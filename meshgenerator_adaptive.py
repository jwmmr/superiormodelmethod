from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "mesh"
AUTO_RESIZE_TARGET_UTILIZATION = 0.97
ASSET_MAP_PATH = SCRIPT_DIR / "roblox_asset_ids.json"
EXPORTER_SCRIPT = SCRIPT_DIR / "roblox_model_export.py"


@dataclass
class Block:
    x: int
    y: int
    w: int
    h: int
    atlas_page: int
    uv_rect: tuple[float, float, float, float]  # u0, v0, u1, v1 in top-left image space
    rot180: bool


def is_fully_transparent(block: np.ndarray) -> bool:
    return bool(np.all(block[:, :, 3] == 0))


def bleed_rgb_into_transparent(rgba: np.ndarray, passes: int = 8) -> np.ndarray:
    """Fill fully transparent RGB with nearby visible colors to reduce dark alpha fringes."""
    out = rgba.copy()
    alpha = out[:, :, 3]
    if np.all(alpha > 0):
        return out

    for _ in range(max(0, passes)):
        transparent = alpha == 0
        if not np.any(transparent):
            break

        grown = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src_y0 = max(0, -dy)
            src_y1 = out.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = out.shape[1] - max(0, dx)

            dst_y0 = max(0, dy)
            dst_y1 = out.shape[0] - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = out.shape[1] - max(0, -dx)

            src_alpha = alpha[src_y0:src_y1, src_x0:src_x1]
            dst_alpha = alpha[dst_y0:dst_y1, dst_x0:dst_x1]
            can_copy = (dst_alpha == 0) & (src_alpha > 0)
            if not np.any(can_copy):
                continue

            dst_rgb = out[dst_y0:dst_y1, dst_x0:dst_x1, :3]
            src_rgb = out[src_y0:src_y1, src_x0:src_x1, :3]
            dst_rgb[can_copy] = src_rgb[can_copy]
            grown = True

        if not grown:
            break

    return out


def make_patch(
    block: np.ndarray,
    patch_size: int,
    edge_bleed_passes: int,
    alpha_feather: float,
) -> np.ndarray:
    """Resize block with alpha-aware filtering to avoid dark edges around transparency."""
    src = bleed_rgb_into_transparent(block, passes=edge_bleed_passes)

    # Premultiply before filtering, then unpremultiply to avoid black fringe artifacts.
    rgb = src[:, :, :3].astype(np.float32)
    a = (src[:, :, 3:4].astype(np.float32)) / 255.0
    rgb_pm = rgb * a

    rgb_img = Image.fromarray(np.clip(rgb_pm, 0, 255).astype(np.uint8), mode="RGB")
    a_img = Image.fromarray(src[:, :, 3], mode="L")

    rgb_resized = np.array(
        rgb_img.resize((patch_size, patch_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    a_resized = np.array(
        a_img.resize((patch_size, patch_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    if alpha_feather > 0:
        a_resized = np.array(
            Image.fromarray(np.clip(a_resized, 0, 255).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(radius=alpha_feather)
            ),
            dtype=np.float32,
        )

    a_norm = np.clip(a_resized / 255.0, 1e-6, 1.0)
    rgb_unpm = np.where(
        a_resized[:, :, None] > 0,
        rgb_resized / a_norm[:, :, None],
        rgb_resized,
    )

    out = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(rgb_unpm, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(a_resized, 0, 255).astype(np.uint8)
    return out


def copy_with_padding(dst: np.ndarray, x: int, y: int, patch: np.ndarray, pad: int) -> None:
    """Place a patch and replicate edge texels into the padding border."""
    ph, pw, _ = patch.shape
    x0 = x + pad
    y0 = y + pad
    dst[y0:y0 + ph, x0:x0 + pw] = patch

    if pad <= 0:
        return

    # Top and bottom padding
    dst[y:y0, x0:x0 + pw] = patch[0:1, :, :]
    dst[y0 + ph:y0 + ph + pad, x0:x0 + pw] = patch[ph - 1:ph, :, :]

    # Left and right padding (including already-padded top/bottom rows)
    dst[y:y0 + ph + pad, x:x0] = dst[y:y0 + ph + pad, x0:x0 + 1]
    dst[y:y0 + ph + pad, x0 + pw:x0 + pw + pad] = dst[y:y0 + ph + pad, x0 + pw - 1:x0 + pw]


def estimate_block_count(width: int, height: int, block_size: int) -> int:
    return math.ceil(width / block_size) * math.ceil(height / block_size)


def optimize_resize_for_atlas(
    src_w: int,
    src_h: int,
    block_size: int,
    patch_size: int,
    atlas_size: int,
    atlas_padding: int,
    target_pages: int,
    target_utilization: float = AUTO_RESIZE_TARGET_UTILIZATION,
) -> tuple[int, int, bool, int, int]:
    """Downscale large inputs so estimated block count fits near one full atlas page."""
    cell = patch_size + (atlas_padding * 2)
    cells_per_row = atlas_size // cell
    pages = max(1, target_pages)
    capacity = max(1, cells_per_row * cells_per_row * pages)

    src_blocks = estimate_block_count(src_w, src_h, block_size)
    if src_blocks <= capacity:
        return src_w, src_h, False, src_blocks, capacity

    target_blocks = max(1, int(capacity * max(0.1, min(1.0, target_utilization))))
    scale = math.sqrt(target_blocks / float(src_blocks))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    # Nudge down if ceil behavior still overshoots atlas capacity.
    while estimate_block_count(new_w, new_h, block_size) > capacity and (new_w > 1 or new_h > 1):
        new_w = max(1, int(math.floor(new_w * 0.995)))
        new_h = max(1, int(math.floor(new_h * 0.995)))

    return new_w, new_h, (new_w != src_w or new_h != src_h), src_blocks, capacity


def build_patch_atlas_and_blocks(
    input_path: Path,
    output_dir: Path,
    block_size: int,
    patch_size: int,
    atlas_size: int,
    atlas_padding: int,
    scramble_seed: int,
    scramble_atlas: bool,
    dedupe_patches: bool,
    random_patch_rotation: bool,
    target_pages: int,
    trim_alpha_bounds: bool,
    edge_bleed_passes: int,
    alpha_feather: float,
    edge_refine_size: int,
    alpha_cutoff: int,
) -> tuple[list[Path], list[Block], tuple[int, int], dict[str, int]]:
    img = Image.open(input_path).convert("RGBA")
    src_w, src_h = img.size

    new_w, new_h, resized, src_blocks, atlas_capacity = optimize_resize_for_atlas(
        src_w=src_w,
        src_h=src_h,
        block_size=block_size,
        patch_size=patch_size,
        atlas_size=atlas_size,
        atlas_padding=atlas_padding,
        target_pages=target_pages,
    )
    if resized:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pixels = np.array(img, dtype=np.uint8)
    h, w, _ = pixels.shape

    blocks_raw: list[tuple[int, int, int, int, int, bool]] = []
    patch_list: list[np.ndarray] = []
    dedupe_index: dict[bytes, int] = {}
    obf_rng = random.Random(scramble_seed ^ 0xA5A5)

    def emit_block(global_x: int, global_y: int, block: np.ndarray) -> None:
        if block.size == 0:
            return
        if int(np.max(block[:, :, 3])) < alpha_cutoff:
            return

        local_x = 0
        local_y = 0
        if trim_alpha_bounds:
            ys, xs = np.where(block[:, :, 3] >= alpha_cutoff)
            if ys.size == 0 or xs.size == 0:
                return
            local_x = int(xs.min())
            local_y = int(ys.min())
            local_x1 = int(xs.max()) + 1
            local_y1 = int(ys.max()) + 1
            block = block[local_y:local_y1, local_x:local_x1]

        patch = make_patch(
            block,
            patch_size=patch_size,
            edge_bleed_passes=edge_bleed_passes,
            alpha_feather=alpha_feather,
        )

        rot180 = bool(random_patch_rotation and (obf_rng.random() < 0.5))
        if rot180:
            patch = np.rot90(patch, 2)

        if dedupe_patches:
            key = patch.tobytes()
            patch_id = dedupe_index.get(key)
            if patch_id is None:
                patch_id = len(patch_list)
                patch_list.append(patch)
                dedupe_index[key] = patch_id
        else:
            patch_id = len(patch_list)
            patch_list.append(patch)

        blocks_raw.append((global_x + local_x, global_y + local_y, block.shape[1], block.shape[0], patch_id, rot180))

    edge_blocks = 0
    refined_sub_blocks = 0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            x1 = min(x + block_size, w)
            y1 = min(y + block_size, h)
            block = pixels[y:y1, x:x1]

            alpha = block[:, :, 3]
            max_a = int(np.max(alpha))
            min_a = int(np.min(alpha))
            if max_a < alpha_cutoff:
                continue

            mixed_alpha = min_a < alpha_cutoff <= max_a
            if mixed_alpha and edge_refine_size < block_size:
                edge_blocks += 1
                step = edge_refine_size
                for sy in range(0, block.shape[0], step):
                    for sx in range(0, block.shape[1], step):
                        sub = block[sy:min(sy + step, block.shape[0]), sx:min(sx + step, block.shape[1])]
                        emit_block(x + sx, y + sy, sub)
                        refined_sub_blocks += 1
            else:
                emit_block(x, y, block)

    if not blocks_raw:
        raise ValueError("Input image appears fully transparent; no mesh blocks were generated.")

    cell = patch_size + (atlas_padding * 2)
    cells_per_row = atlas_size // cell
    if cells_per_row <= 0:
        raise ValueError("Atlas is too small for the requested patch_size + padding.")

    max_patches_per_page = cells_per_row * cells_per_row
    if max_patches_per_page <= 0:
        raise ValueError("Invalid atlas packing settings.")

    patch_ids = list(range(len(patch_list)))
    if scramble_atlas:
        rng = random.Random(scramble_seed)
        rng.shuffle(patch_ids)

    patch_location: dict[int, tuple[int, int, int]] = {}
    pages = math.ceil(len(patch_ids) / max_patches_per_page)
    atlas_images = [np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8) for _ in range(pages)]

    for idx, patch_id in enumerate(patch_ids):
        page = idx // max_patches_per_page
        local_idx = idx % max_patches_per_page
        col = local_idx % cells_per_row
        row = local_idx // cells_per_row
        x = col * cell
        y = row * cell

        copy_with_padding(atlas_images[page], x, y, patch_list[patch_id], atlas_padding)
        patch_location[patch_id] = (page, x, y)

    atlas_paths: list[Path] = []
    for page, data in enumerate(atlas_images):
        atlas_path = output_dir / f"patch_atlas_{page}.png"
        Image.fromarray(data, mode="RGBA").save(atlas_path)
        atlas_paths.append(atlas_path)

    blocks: list[Block] = []
    for x, y, bw, bh, patch_id, rot180 in blocks_raw:
        page, px, py = patch_location[patch_id]
        u0 = (px + atlas_padding) / atlas_size
        v0 = (py + atlas_padding) / atlas_size
        u1 = (px + atlas_padding + patch_size) / atlas_size
        v1 = (py + atlas_padding + patch_size) / atlas_size
        blocks.append(Block(x=x, y=y, w=bw, h=bh, atlas_page=page, uv_rect=(u0, v0, u1, v1), rot180=rot180))

    stats = {
        "source_input_width": src_w,
        "source_input_height": src_h,
        "auto_resized": resized,
        "source_estimated_blocks": src_blocks,
        "atlas_block_capacity_per_page": atlas_capacity,
        "auto_resize_target_utilization": AUTO_RESIZE_TARGET_UTILIZATION,
        "input_width": w,
        "input_height": h,
        "non_transparent_blocks": len(blocks),
        "unique_patches": len(patch_list),
        "atlas_pages": pages,
        "edge_blocks_refined": edge_blocks,
        "refined_sub_blocks_emitted": refined_sub_blocks,
    }
    return atlas_paths, blocks, (w, h), stats


def write_mtl(mtl_path: Path, atlas_paths: list[Path]) -> None:
    with mtl_path.open("w", encoding="utf-8") as f:
        for i, atlas_path in enumerate(atlas_paths):
            f.write(f"newmtl atlas_page_{i}\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ka 0.000000 0.000000 0.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("d 1.0\n")
            f.write("illum 1\n")
            f.write(f"map_Kd {atlas_path.name}\n\n")
            f.write(f"map_d {atlas_path.name}\n\n")


def write_single_page_mtl(mtl_path: Path, atlas_path: Path, page: int) -> str:
    mat_name = f"atlas_page_{page}"
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write(f"newmtl {mat_name}\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ka 0.000000 0.000000 0.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {atlas_path.name}\n")
        f.write(f"map_d {atlas_path.name}\n")
    return mat_name


def write_mesh_chunks(
    blocks: list[Block],
    image_dims: tuple[int, int],
    atlas_paths: list[Path],
    atlas_size: int,
    output_dir: Path,
    max_tris_per_mesh: int,
    pixel_scale: float,
    force_mesh_chunks: int | None = None,
) -> list[Path]:
    # Intentionally group by atlas page so all geometry sharing a texture
    # is stored in the same OBJ. Roblox will still split internal meshes as needed.
    if max_tris_per_mesh < 2:
        raise ValueError("max_tris_per_mesh must be at least 2.")

    output_dir.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h = image_dims
    texel = 0.5 / float(atlas_size)

    mesh_paths: list[Path] = []
    by_page: dict[int, list[Block]] = {}
    for b in sorted(blocks, key=lambda blk: (blk.atlas_page, blk.y, blk.x)):
        by_page.setdefault(b.atlas_page, []).append(b)

    for page, page_blocks in sorted(by_page.items()):
        obj_path = output_dir / f"patch_mesh_p{page}.obj"
        mtl_path = output_dir / f"patch_mesh_p{page}.mtl"
        mat_name = write_single_page_mtl(mtl_path, atlas_paths[page], page)

        with obj_path.open("w", encoding="utf-8") as f:
            f.write(f"# Mesh atlas page {page}\n")
            f.write(f"# Source dimensions: {orig_w}x{orig_h}\n")
            f.write(f"mtllib {mtl_path.name}\n")
            f.write(f"usemtl {mat_name}\n")

            v_idx = 1
            uv_idx = 1
            for b in page_blocks:
                x0 = b.x
                y0 = b.y
                x1 = b.x + b.w
                y1 = b.y + b.h

                sx0 = x0 * pixel_scale
                sy0 = y0 * pixel_scale
                sx1 = x1 * pixel_scale
                sy1 = y1 * pixel_scale

                # Vertex positions in source pixel space; Y is flipped for OBJ.
                f.write(f"v {sx0:.6f} {(orig_h * pixel_scale) - sy0:.6f} 0\n")
                f.write(f"v {sx1:.6f} {(orig_h * pixel_scale) - sy0:.6f} 0\n")
                f.write(f"v {sx1:.6f} {(orig_h * pixel_scale) - sy1:.6f} 0\n")
                f.write(f"v {sx0:.6f} {(orig_h * pixel_scale) - sy1:.6f} 0\n")

                u0, v0, u1, v1 = b.uv_rect
                # Pull UVs inward by half a texel to reduce border bleed from filtering/mips.
                u0 += texel
                v0 += texel
                u1 -= texel
                v1 -= texel
                # OBJ V coordinate is bottom-origin, so convert from top-left image UV.
                if b.rot180:
                    # Inverse mapping for 180° patch rotation.
                    f.write(f"vt {u1} {1.0 - v1}\n")
                    f.write(f"vt {u0} {1.0 - v1}\n")
                    f.write(f"vt {u0} {1.0 - v0}\n")
                    f.write(f"vt {u1} {1.0 - v0}\n")
                else:
                    f.write(f"vt {u0} {1.0 - v0}\n")
                    f.write(f"vt {u1} {1.0 - v0}\n")
                    f.write(f"vt {u1} {1.0 - v1}\n")
                    f.write(f"vt {u0} {1.0 - v1}\n")

                f.write(
                    f"f {v_idx}/{uv_idx} {v_idx + 1}/{uv_idx + 1} "
                    f"{v_idx + 2}/{uv_idx + 2} {v_idx + 3}/{uv_idx + 3}\n"
                )
                v_idx += 4
                uv_idx += 4

        mesh_paths.append(obj_path)

    return mesh_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate low-mesh patch-atlas OBJ output from input.png"
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "input.png")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--atlas-size", type=int, default=1024)
    parser.add_argument("--atlas-padding", type=int, default=4)
    parser.add_argument("--max-tris", type=int, default=10000)
    parser.add_argument("--scramble-seed", type=int, default=1337)
    parser.add_argument(
        "--scramble-atlas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize atlas patch placement (obfuscation). Disable for maximum visual stability.",
    )
    parser.add_argument(
        "--random-patch-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly rotate atlas patches by 180 degrees and compensate UVs for stronger obfuscation.",
    )
    parser.add_argument(
        "--dedupe-patches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse identical patch textures to reduce atlas size; may reduce local fidelity.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=0.01,
        help="World-space size per source pixel. Lower value makes output mesh smaller.",
    )
    parser.add_argument(
        "--trim-alpha-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim each block to the tightest non-transparent bounds before meshing.",
    )
    parser.add_argument(
        "--edge-bleed-passes",
        type=int,
        default=24,
        help="How aggressively to propagate RGB into transparent texels.",
    )
    parser.add_argument(
        "--alpha-feather",
        type=float,
        default=0.75,
        help="Small alpha blur in patch space for smoother anti-aliased edges.",
    )
    parser.add_argument(
        "--edge-refine-size",
        type=int,
        default=2,
        help="Subdivide mixed-alpha edge tiles to this size (in source pixels).",
    )
    parser.add_argument(
        "--alpha-cutoff",
        type=int,
        default=8,
        help="Ignore texels below this alpha when trimming/refining to reduce dark fringe blocks.",
    )
    parser.add_argument(
        "--skip-exporter",
        action="store_true",
        help="Generate mesh outputs only and skip asset prompts/export step.",
    )
    return parser.parse_args()


def prepare_clean_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return output_dir


def resolve_input_image(input_path: Path, output_dir: Path) -> Path:
    """Resolve input image path, prompting user when default file is missing."""
    if input_path.exists():
        return input_path

    user_value = input("Drop image file or paste path here:").strip().strip('"')
    if not user_value:
        raise FileNotFoundError("No input image provided.")

    parsed = urllib.parse.urlparse(user_value)
    if parsed.scheme in ("http", "https"):
        suffix = Path(parsed.path).suffix or ".png"
        downloaded = output_dir / f"downloaded_input{suffix}"
        urllib.request.urlretrieve(user_value, downloaded)
        print(f"Downloaded input image to {downloaded}")
        return downloaded

    local_path = Path(user_value)
    if not local_path.is_absolute():
        local_path = (SCRIPT_DIR / local_path).resolve()

    if not local_path.exists():
        raise FileNotFoundError(f"Input image not found: {local_path}")
    return local_path


def prompt_high_quality_mode() -> bool:
    value = input("Higher quality result? (y/N):").strip().lower()
    return value in ("y", "yes")


def _parse_asset_id_input(value: str):
    text = value.strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    return text


def _prompt_ids_in_order(items: list[str], section_name: str, existing: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    total = len(items)
    if total == 0:
        return result

    print(f"\nEnter {section_name} asset IDs in order ({total} total).")
    print("Tip: paste numeric IDs or full rbxassetid://... values.")
    for i, name in enumerate(items, start=1):
        current = existing.get(name, 0)
        raw = input(f"[{i}/{total}] {name} (current: {current}) -> ").strip()
        result[name] = _parse_asset_id_input(raw) if raw else current
    return result


def prompt_and_update_asset_map(manifest: dict, asset_map_path: Path = ASSET_MAP_PATH) -> Path:
    mesh_files: list[str] = manifest.get("mesh_files", [])
    atlas_files: list[str] = manifest.get("atlas_files", [])

    existing = {
        "mesh_ids": {},
        "texture_ids": {},
        "notes": "Replace 0 with uploaded Roblox asset IDs.",
    }
    if asset_map_path.exists():
        try:
            existing = json.loads(asset_map_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    updated = {
        "mesh_ids": _prompt_ids_in_order(mesh_files, "mesh", existing.get("mesh_ids", {})),
        "texture_ids": _prompt_ids_in_order(atlas_files, "texture", existing.get("texture_ids", {})),
        "notes": existing.get("notes", "Replace 0 with uploaded Roblox asset IDs."),
    }
    asset_map_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"Updated asset IDs: {asset_map_path}")
    return asset_map_path


def run_exporter(manifest_path: Path, asset_map_path: Path) -> None:
    if not EXPORTER_SCRIPT.exists():
        raise FileNotFoundError(f"Exporter script not found: {EXPORTER_SCRIPT}")

    cmd = [
        sys.executable,
        str(EXPORTER_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--asset-map",
        str(asset_map_path),
    ]
    print("\nRunning Roblox exporter...")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_dir = prepare_clean_output_dir(args.output_dir)
    input_path = resolve_input_image(args.input, output_dir)
    high_quality = prompt_high_quality_mode()

    if args.block_size <= 0:
        raise ValueError("block-size must be > 0")
    if args.patch_size <= 0:
        raise ValueError("patch-size must be > 0")
    if args.patch_size > args.block_size:
        raise ValueError("patch-size should be <= block-size for sensible downscaling")
    if args.pixel_scale <= 0:
        raise ValueError("pixel-scale must be > 0")
    if args.edge_refine_size <= 0:
        raise ValueError("edge-refine-size must be > 0")
    if args.edge_refine_size > args.block_size:
        raise ValueError("edge-refine-size should be <= block-size")
    if args.alpha_cutoff < 1 or args.alpha_cutoff > 255:
        raise ValueError("alpha-cutoff must be in 1..255")

    print("Building patch atlas and block map...")
    atlas_paths, blocks, image_dims, stats = build_patch_atlas_and_blocks(
        input_path=input_path,
        output_dir=output_dir,
        block_size=args.block_size,
        patch_size=args.patch_size,
        atlas_size=args.atlas_size,
        atlas_padding=args.atlas_padding,
        scramble_seed=args.scramble_seed,
        scramble_atlas=args.scramble_atlas,
        dedupe_patches=args.dedupe_patches,
        random_patch_rotation=args.random_patch_rotation,
        target_pages=2 if high_quality else 1,
        trim_alpha_bounds=args.trim_alpha_bounds,
        edge_bleed_passes=args.edge_bleed_passes,
        alpha_feather=args.alpha_feather,
        edge_refine_size=args.edge_refine_size,
        alpha_cutoff=args.alpha_cutoff,
    )

    print("Writing mesh chunks...")
    mesh_paths = write_mesh_chunks(
        blocks=blocks,
        image_dims=image_dims,
        atlas_paths=atlas_paths,
        atlas_size=args.atlas_size,
        output_dir=output_dir,
        max_tris_per_mesh=args.max_tris,
        pixel_scale=args.pixel_scale,
        force_mesh_chunks=2 if high_quality else None,
    )

    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "config": {
            "block_size": args.block_size,
            "patch_size": args.patch_size,
            "atlas_size": args.atlas_size,
            "atlas_padding": args.atlas_padding,
            "max_tris": args.max_tris,
            "scramble_seed": args.scramble_seed,
            "scramble_atlas": args.scramble_atlas,
            "dedupe_patches": args.dedupe_patches,
            "random_patch_rotation": args.random_patch_rotation,
            "high_quality_mode": high_quality,
            "target_atlas_pages": 2 if high_quality else 1,
            "pixel_scale": args.pixel_scale,
            "trim_alpha_bounds": args.trim_alpha_bounds,
            "edge_bleed_passes": args.edge_bleed_passes,
            "alpha_feather": args.alpha_feather,
            "edge_refine_size": args.edge_refine_size,
            "alpha_cutoff": args.alpha_cutoff,
        },
        "stats": stats,
        "atlas_files": [p.name for p in atlas_paths],
        "mesh_files": [p.name for p in mesh_paths],
    }

    manifest_path = output_dir / "patch_mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Wrote {len(mesh_paths)} mesh file(s), {len(atlas_paths)} atlas page(s).")
    print(f"Manifest: {manifest_path}")

    if not args.skip_exporter:
        asset_map_path = prompt_and_update_asset_map(manifest, ASSET_MAP_PATH)
        run_exporter(manifest_path, asset_map_path)


if __name__ == "__main__":
    main()
