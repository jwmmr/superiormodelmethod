from PIL import Image
import numpy as np
from collections import defaultdict
import math
import os
from pathlib import Path          # ← NEW

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Path to the folder where this script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Image we want to process (must sit next to the .py file)
input_image = SCRIPT_DIR / "input.png"        # ← BULLET‑PROOF

# Where to drop outputs (same folder; change if you like)
atlas_output  = SCRIPT_DIR / "color_atlas.png"
mesh_out_dir  = SCRIPT_DIR                    # could be another Path
# ------------------------------------------------------------------


def generate_color_atlas(input_path, output_path=atlas_output):
    """Generate a 1024×1024 texture with all unique colors including partial transparency"""
    orig_img = Image.open(input_path).convert('RGBA')  # Force RGBA for alpha handling
    orig_pixels = np.array(orig_img)
    h, w, _ = orig_pixels.shape

    # Collect all unique colors with some visibility (alpha > 0)
    unique_colors = {
        tuple(orig_pixels[y, x])
        for y in range(h)
        for x in range(w)
        if orig_pixels[y, x, 3] > 0     # keep pixels with alpha > 0
    }

    num_colors = len(unique_colors)
    print(f"Found {num_colors} visible colors")

    tex_size   = 1024
    max_colors = tex_size * tex_size
    unique_colors = list(unique_colors)[:max_colors]  # clamp if needed

    # Grid size
    cols = math.ceil(math.sqrt(len(unique_colors)))
    rows = math.ceil(len(unique_colors) / cols)

    if len(unique_colors) == 0:
        raise ValueError("Image has no non‑transparent pixels.")

    # Build the atlas
    texture     = np.zeros((tex_size, tex_size, 4), dtype=np.uint8)
    color_to_uv = {}
    cell_w      = tex_size / cols
    cell_h      = tex_size / rows

    for i, color in enumerate(unique_colors):
        col = i % cols
        row = i // cols
        x0, x1 = int(col * cell_w),   int((col + 1) * cell_w)
        y0, y1 = int(row * cell_h),   int((row + 1) * cell_h)
        texture[y0:y1, x0:x1] = color

        # UV coordinate (centre of the cell)
        u = (x0 + x1) / (2 * tex_size)
        v = (y0 + y1) / (2 * tex_size)
        color_to_uv[color] = (u, v)

    # Fill remaining empty cells with last colour (fully opaque)
    for i in range(len(unique_colors), cols * rows):
        col = i % cols
        row = i // cols
        x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
        y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
        texture[y0:y1, x0:x1] = (*unique_colors[-1][:3], 255)

    Image.fromarray(texture).save(output_path)
    print(f"Color atlas saved to {output_path}")
    return output_path, color_to_uv, (w, h)


def is_tile_fully_transparent(orig_pixels, x0, x1, y0, y1):
    """True if every pixel in the tile is alpha == 0"""
    return np.all(orig_pixels[y0:y1, x0:x1, 3] == 0)


def create_tiled_meshes(input_path, color_atlas_path, color_to_uv, original_dims,
                        max_tris=10000):
    """Create mesh tiles, skipping only 100 % transparent tiles"""
    orig_w, orig_h = original_dims
    orig_pixels    = np.array(Image.open(input_path).convert('RGBA'))

    # Pixels per mesh ⇒ triangles per mesh
    max_pixels_per_mesh = max_tris // 2
    cols_per_mesh       = min(orig_w, math.isqrt(max_pixels_per_mesh))
    rows_per_mesh       = min(orig_h, max_pixels_per_mesh // cols_per_mesh)

    tiles_x = math.ceil(orig_w / cols_per_mesh)
    tiles_y = math.ceil(orig_h / rows_per_mesh)
    print(f"Splitting into {tiles_x} × {tiles_y} tiles")

    mesh_paths, skipped = [], 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * cols_per_mesh
            x1 = min((tx + 1) * cols_per_mesh, orig_w)
            y0 = ty * rows_per_mesh
            y1 = min((ty + 1) * rows_per_mesh, orig_h)

            if is_tile_fully_transparent(orig_pixels, x0, x1, y0, y1):
                skipped += 1
                continue

            mesh_path = mesh_out_dir / f"pixel_mesh_{tx}_{ty}.obj"
            mesh_paths.append(mesh_path)

            with mesh_path.open('w') as f:
                f.write(f"# Tile {tx}, {ty}\n")
                v_idx = uv_idx = 1

                for y in range(y0, y1):
                    for x in range(x0, x1):
                        color = tuple(orig_pixels[y, x])

                        # Quad vertices (flip Y)
                        f.write(f"v {x} {orig_h - y} 0\n")
                        f.write(f"v {x+1} {orig_h - y} 0\n")
                        f.write(f"v {x+1} {orig_h - y - 1} 0\n")
                        f.write(f"v {x} {orig_h - y - 1} 0\n")

                        # Same UV for all 4 vertices
                        u, v = color_to_uv.get(color, (0.0, 0.0))
                        for _ in range(4):
                            f.write(f"vt {u} {1 - v}\n")

                        if color[3] > 0:  # non‑transparent
                            f.write(f"f {v_idx}/{uv_idx} {v_idx+1}/{uv_idx+1} "
                                    f"{v_idx+2}/{uv_idx+2} {v_idx+3}/{uv_idx+3}\n")
                        v_idx += 4
                        uv_idx += 4

    print(f"Generated {len(mesh_paths)} mesh tiles (skipped {skipped} fully transparent tiles)")
    return mesh_paths


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating color atlas …")
    atlas_path, uv_map, orig_dims = generate_color_atlas(input_image)

    print("Creating mesh tiles …")
    mesh_files = create_tiled_meshes(input_image, atlas_path, uv_map, orig_dims)

    print(f"Process complete! Created {len(mesh_files)} mesh files.")
