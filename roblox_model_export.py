from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MESH_RESULTS_DIR = SCRIPT_DIR / "results" / "mesh"
ROBLOX_RESULTS_DIR = SCRIPT_DIR / "results" / "roblox"


MESH_PAGE_RE = re.compile(r"_p(\d+)\.obj$", re.IGNORECASE)

DEFAULT_OBF_MODELS = 8
DEFAULT_OBF_PARTS_PER_MODEL = 546
OBF_TEMPLATE_VERSION = 4


def _ref() -> str:
    return f"RBX{uuid.uuid4().hex.upper()}"


def _add_item(parent: ET.Element, class_name: str) -> ET.Element:
    return ET.SubElement(parent, "Item", {"class": class_name, "referent": _ref()})


def _props(item: ET.Element) -> ET.Element:
    return ET.SubElement(item, "Properties")


def _add_string(props: ET.Element, name: str, value: str) -> None:
    node = ET.SubElement(props, "string", {"name": name})
    node.text = value


def _add_bool(props: ET.Element, name: str, value: bool) -> None:
    node = ET.SubElement(props, "bool", {"name": name})
    node.text = "true" if value else "false"


def _add_float(props: ET.Element, name: str, value: float) -> None:
    node = ET.SubElement(props, "float", {"name": name})
    node.text = f"{value:.6f}"


def _add_token(props: ET.Element, name: str, value: int) -> None:
    node = ET.SubElement(props, "token", {"name": name})
    node.text = str(value)


def _set_smooth_surfaces(props: ET.Element) -> None:
    """Use smooth surfaces to prevent automatic Snap joints in Studio."""
    for name in ("TopSurface", "BottomSurface", "LeftSurface", "RightSurface", "FrontSurface", "BackSurface"):
        _add_token(props, name, 0)


def _add_content(props: ET.Element, name: str, value: str) -> None:
    node = ET.SubElement(props, "Content", {"name": name})
    url = ET.SubElement(node, "url")
    url.text = value


def _add_vector3(props: ET.Element, name: str, x: float, y: float, z: float) -> None:
    node = ET.SubElement(props, "Vector3", {"name": name})
    nx = ET.SubElement(node, "X")
    ny = ET.SubElement(node, "Y")
    nz = ET.SubElement(node, "Z")
    nx.text = f"{x:.6f}"
    ny.text = f"{y:.6f}"
    nz.text = f"{z:.6f}"


def _add_color3(props: ET.Element, name: str, r255: int, g255: int, b255: int) -> None:
    node = ET.SubElement(props, "Color3", {"name": name})
    nr = ET.SubElement(node, "R")
    ng = ET.SubElement(node, "G")
    nb = ET.SubElement(node, "B")
    nr.text = f"{max(0, min(255, r255)) / 255.0:.6f}"
    ng.text = f"{max(0, min(255, g255)) / 255.0:.6f}"
    nb.text = f"{max(0, min(255, b255)) / 255.0:.6f}"


def _add_cframe_identity(props: ET.Element, name: str, x: float, y: float, z: float) -> None:
    node = ET.SubElement(props, "CoordinateFrame", {"name": name})
    ex = ET.SubElement(node, "X")
    ey = ET.SubElement(node, "Y")
    ez = ET.SubElement(node, "Z")
    r00 = ET.SubElement(node, "R00")
    r01 = ET.SubElement(node, "R01")
    r02 = ET.SubElement(node, "R02")
    r10 = ET.SubElement(node, "R10")
    r11 = ET.SubElement(node, "R11")
    r12 = ET.SubElement(node, "R12")
    r20 = ET.SubElement(node, "R20")
    r21 = ET.SubElement(node, "R21")
    r22 = ET.SubElement(node, "R22")

    ex.text = f"{x:.6f}"
    ey.text = f"{y:.6f}"
    ez.text = f"{z:.6f}"

    # Identity rotation; for Camera this points down -Z in Roblox.
    r00.text = "1"
    r01.text = "0"
    r02.text = "0"
    r10.text = "0"
    r11.text = "1"
    r12.text = "0"
    r20.text = "0"
    r21.text = "0"
    r22.text = "1"


def _v_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _v_norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]))
    if mag < 1e-9:
        return (0.0, 0.0, -1.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _add_cframe_look_at(
    props: ET.Element,
    name: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    """Write a Roblox CoordinateFrame that looks from eye -> target."""
    look = _v_norm(_v_sub(target, eye))
    up = (0.0, 1.0, 0.0)
    right = _v_norm(_v_cross(look, up))
    if abs(right[0]) < 1e-9 and abs(right[1]) < 1e-9 and abs(right[2]) < 1e-9:
        up = (0.0, 0.0, 1.0)
        right = _v_norm(_v_cross(look, up))
    true_up = _v_cross(right, look)

    node = ET.SubElement(props, "CoordinateFrame", {"name": name})
    ex = ET.SubElement(node, "X")
    ey = ET.SubElement(node, "Y")
    ez = ET.SubElement(node, "Z")
    r00 = ET.SubElement(node, "R00")
    r01 = ET.SubElement(node, "R01")
    r02 = ET.SubElement(node, "R02")
    r10 = ET.SubElement(node, "R10")
    r11 = ET.SubElement(node, "R11")
    r12 = ET.SubElement(node, "R12")
    r20 = ET.SubElement(node, "R20")
    r21 = ET.SubElement(node, "R21")
    r22 = ET.SubElement(node, "R22")

    ex.text = f"{eye[0]:.6f}"
    ey.text = f"{eye[1]:.6f}"
    ez.text = f"{eye[2]:.6f}"

    # Roblox CFrame basis columns: right, up, back (-look)
    r00.text = f"{right[0]:.9f}"
    r01.text = f"{true_up[0]:.9f}"
    r02.text = f"{-look[0]:.9f}"
    r10.text = f"{right[1]:.9f}"
    r11.text = f"{true_up[1]:.9f}"
    r12.text = f"{-look[1]:.9f}"
    r20.text = f"{right[2]:.9f}"
    r21.text = f"{true_up[2]:.9f}"
    r22.text = f"{-look[2]:.9f}"


def normalize_asset_id(value: str | int | None, fallback: str = "rbxassetid://0") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    if text.startswith("rbxassetid://"):
        return text
    if text.isdigit():
        return f"rbxassetid://{text}"
    return text


def parse_page_from_mesh_name(mesh_name: str) -> int:
    match = MESH_PAGE_RE.search(mesh_name)
    if not match:
        return 0
    return int(match.group(1))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_size(v: float) -> float:
    return max(0.05, float(v))


def read_obj_bounds(obj_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not obj_path.exists():
        return None

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    found = False

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                continue

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
            found = True

    if not found:
        return None
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def build_mesh_placement_map(manifest: dict) -> dict[str, dict[str, float]]:
    output_dir = Path(manifest.get("output_dir", SCRIPT_DIR))
    mesh_files: list[str] = manifest.get("mesh_files", [])
    placement: dict[str, dict[str, float]] = {}

    for mesh_file in mesh_files:
        bounds = read_obj_bounds(output_dir / mesh_file)
        if bounds is None:
            placement[mesh_file] = {
                "cx": 0.0,
                "cy": 0.0,
                "cz": 0.0,
                "sx": 1.0,
                "sy": 1.0,
                "sz": 0.05,
            }
            continue

        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        placement[mesh_file] = {
            "cx": (min_x + max_x) * 0.5,
            "cy": (min_y + max_y) * 0.5,
            "cz": (min_z + max_z) * 0.5,
            "sx": _safe_size(max_x - min_x),
            "sy": _safe_size(max_y - min_y),
            "sz": _safe_size(max_z - min_z),
        }

    return placement


def ensure_asset_map(path: Path, manifest: dict) -> dict:
    mesh_files = manifest.get("mesh_files", [])
    atlas_files = manifest.get("atlas_files", [])

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    template = {
        "mesh_ids": {name: 0 for name in mesh_files},
        "texture_ids": {name: 0 for name in atlas_files},
        "notes": "Replace 0 with uploaded Roblox asset IDs.",
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template


def ensure_adjustments_map(path: Path, manifest: dict) -> dict:
    mesh_files = manifest.get("mesh_files", [])

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    template = {
        "global": {
            "dx_per_index": 0.0,
            "dy_per_index": 0.0,
            "dz_per_index": 0.0,
        },
        "per_mesh": {
            name: {"dx": 0.0, "dy": 0.0, "dz": 0.0} for name in mesh_files
        },
        "notes": "Use tiny values (e.g. 0.001 to 0.02) to correct drift/tearing between meshes.",
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template


def ensure_alignment_samples(path: Path, manifest: dict) -> dict:
    mesh_files = manifest.get("mesh_files", [])

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    template = {
        "mode": "index-interpolate",
        "samples": {
            name: {"target": [0.0, 0.0, 0.0], "enabled": False}
            for name in mesh_files[:6]
        },
        "notes": "Set enabled=true and target=[x,y,z] for a few known-correct mesh positions.",
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template


def ensure_obf_template(path: Path, model_count: int, parts_per_model: int, seed: int) -> dict:
    """Create/load reusable normalized obfuscation cloud points."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if int(existing.get("version", 0)) == OBF_TEMPLATE_VERSION:
            return existing

    rng = random.Random(seed)
    models: list[dict] = []

    for _ in range(model_count):
        points = []
        for _ in range(parts_per_model):
            # Full-canvas random coverage per model; each model gets its own randomized layout.
            px = rng.uniform(0.0, 1.0)
            py = rng.uniform(0.0, 1.0)
            pz = rng.uniform(-1.0, 1.0)
            pr = rng.randint(0, 255)
            pg = rng.randint(0, 255)
            pb = rng.randint(0, 255)
            points.append({"x": px, "y": py, "z": pz, "r": pr, "g": pg, "b": pb})
        models.append({"name": "obf", "points": points})

    template = {
        "version": OBF_TEMPLATE_VERSION,
        "models": models,
        "notes": "Version 4: unique per-model random points plus per-part random color.",
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template


def _lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def fit_linear_1d(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit y = a + b*x with least squares. Returns (a, b)."""
    if not points:
        return 0.0, 0.0
    n = float(len(points))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = (n * sxx) - (sx * sx)
    if abs(denom) < 1e-9:
        return sy / n, 0.0
    b = ((n * sxy) - (sx * sy)) / denom
    a = (sy - (b * sx)) / n
    return a, b


def build_interpolated_alignment(
    mesh_files: list[str],
    base_positions: dict[str, tuple[float, float, float]],
    samples_cfg: dict,
) -> dict[str, tuple[float, float, float]]:
    samples_raw = samples_cfg.get("samples", {})
    sample_points: list[tuple[int, float, float, float]] = []

    for idx, name in enumerate(mesh_files):
        sample = samples_raw.get(name)
        if not isinstance(sample, dict):
            continue
        if not bool(sample.get("enabled", False)):
            continue
        target = sample.get("target")
        if not isinstance(target, list) or len(target) < 3:
            continue
        bx, by, bz = base_positions.get(name, (0.0, 0.0, 0.0))
        try:
            dx = float(target[0]) - bx
            dy = float(target[1]) - by
            dz = float(target[2]) - bz
        except (TypeError, ValueError):
            continue
        sample_points.append((idx, dx, dy, dz))

    if not sample_points:
        return {name: (0.0, 0.0, 0.0) for name in mesh_files}

    sample_points.sort(key=lambda p: p[0])

    if len(sample_points) == 1:
        _, dx, dy, dz = sample_points[0]
        return {name: (dx, dy, dz) for name in mesh_files}

    result: dict[str, tuple[float, float, float]] = {}
    for idx, name in enumerate(mesh_files):
        if idx <= sample_points[0][0]:
            i0, dx0, dy0, dz0 = sample_points[0]
            i1, dx1, dy1, dz1 = sample_points[1]
            span = max(1, i1 - i0)
            t = (idx - i0) / span
            result[name] = (_lerp(dx0, dx1, t), _lerp(dy0, dy1, t), _lerp(dz0, dz1, t))
            continue

        if idx >= sample_points[-1][0]:
            i0, dx0, dy0, dz0 = sample_points[-2]
            i1, dx1, dy1, dz1 = sample_points[-1]
            span = max(1, i1 - i0)
            t = (idx - i0) / span
            result[name] = (_lerp(dx0, dx1, t), _lerp(dy0, dy1, t), _lerp(dz0, dz1, t))
            continue

        for p in range(len(sample_points) - 1):
            i0, dx0, dy0, dz0 = sample_points[p]
            i1, dx1, dy1, dz1 = sample_points[p + 1]
            if i0 <= idx <= i1:
                span = max(1, i1 - i0)
                t = (idx - i0) / span
                result[name] = (_lerp(dx0, dx1, t), _lerp(dy0, dy1, t), _lerp(dz0, dz1, t))
                break

    return result


def build_size_linear_alignment(
    mesh_files: list[str],
    placement: dict[str, dict[str, float]],
    base_positions: dict[str, tuple[float, float, float]],
    samples_cfg: dict,
) -> dict[str, tuple[float, float, float]]:
    samples_raw = samples_cfg.get("samples", {})

    x_points: list[tuple[float, float]] = []
    y_points: list[tuple[float, float]] = []
    z_points: list[tuple[float, float]] = []

    for name in mesh_files:
        sample = samples_raw.get(name)
        if not isinstance(sample, dict) or not bool(sample.get("enabled", False)):
            continue
        target = sample.get("target")
        if not isinstance(target, list) or len(target) < 3:
            continue

        bx, by, bz = base_positions.get(name, (0.0, 0.0, 0.0))
        place = placement.get(name, {"sx": 1.0, "sy": 1.0, "sz": 1.0})
        try:
            dx = float(target[0]) - bx
            dy = float(target[1]) - by
            dz = float(target[2]) - bz
            sx = float(place.get("sx", 1.0))
            sy = float(place.get("sy", 1.0))
            sz = float(place.get("sz", 1.0))
        except (TypeError, ValueError):
            continue

        x_points.append((sx, dx))
        y_points.append((sy, dy))
        z_points.append((sz, dz))

    ax, bx = fit_linear_1d(x_points)
    ay, by = fit_linear_1d(y_points)
    az, bz = fit_linear_1d(z_points)

    result: dict[str, tuple[float, float, float]] = {}
    for name in mesh_files:
        place = placement.get(name, {"sx": 1.0, "sy": 1.0, "sz": 1.0})
        sx = float(place.get("sx", 1.0))
        sy = float(place.get("sy", 1.0))
        sz = float(place.get("sz", 1.0))
        result[name] = (
            ax + (bx * sx),
            ay + (by * sy),
            az + (bz * sz),
        )
    return result


def add_obf_models(
    parent_model: ET.Element,
    image_w: float,
    image_h: float,
    obf_template: dict,
    z_center: float = 0.0,
    z_span: float = 0.12,
) -> None:
    """Instantiate reusable obfuscation clouds using only Part-based holders."""
    models = obf_template.get("models", [])
    root_holder: ET.Element | None = None
    for idx, model in enumerate(models):
        holder_parent = parent_model if idx == 0 else (root_holder if root_holder is not None else parent_model)
        holder = _add_item(holder_parent, "Part")
        if idx == 0:
            root_holder = holder
        hp = _props(holder)
        _add_string(hp, "Name", "obf")
        _add_bool(hp, "Anchored", False)
        _add_bool(hp, "CanCollide", True)
        _add_bool(hp, "CanQuery", True)
        _add_bool(hp, "CanTouch", True)
        _add_bool(hp, "Locked", False)
        _add_float(hp, "Transparency", 1.0)
        _add_token(hp, "Material", 256)
        _set_smooth_surfaces(hp)
        _add_vector3(hp, "Size", 1.0, 1.0, 1.0)
        _add_cframe_identity(hp, "CFrame", image_w * 0.5, image_h * 0.5, z_center)

        points = model.get("points", []) if isinstance(model, dict) else []
        for p in points:
            try:
                nx = float(p.get("x", 0.5))
                ny = float(p.get("y", 0.5))
                nz = float(p.get("z", 0.0))
                pr = int(p.get("r", 255))
                pg = int(p.get("g", 255))
                pb = int(p.get("b", 255))
            except (TypeError, ValueError, AttributeError):
                nx, ny, nz = 0.5, 0.5, 0.0
                pr, pg, pb = 255, 255, 255

            x = nx * image_w
            y = ny * image_h
            z = z_center + (nz * z_span * 0.5)

            part = _add_item(holder, "Part")
            pp = _props(part)
            _add_string(pp, "Name", "Part")
            _add_bool(pp, "Anchored", False)
            _add_bool(pp, "CanCollide", True)
            _add_bool(pp, "CanQuery", True)
            _add_bool(pp, "CanTouch", True)
            _add_bool(pp, "Locked", False)
            _add_float(pp, "Transparency", 1.0)
            _add_token(pp, "Material", 256)
            _set_smooth_surfaces(pp)
            _add_color3(pp, "Color", pr, pg, pb)
            _add_vector3(pp, "Size", 4.0, 1.0, 2.0)
            _add_cframe_identity(pp, "CFrame", x, y, z)


def compute_camera(
    image_w: float,
    image_h: float,
    fov_deg: float,
    margin: float,
    viewport_aspect: float,
    fit_padding: float,
    distance_scale: float,
    safe_epsilon: float,
    fit_mode: str,
    center_x: float | None = None,
    center_y: float | None = None,
) -> tuple[tuple[float, float, float], float]:
    cx = (image_w / 2.0) if center_x is None else float(center_x)
    cy = (image_h / 2.0) if center_y is None else float(center_y)

    # Use the viewing viewport aspect. In contain mode we clamp to <= 1.0 for conservative width fit.
    aspect_raw = max(1e-6, float(viewport_aspect))
    mode = fit_mode.strip().lower()
    if mode == "contain":
        # Conservative default for unknown Roblox capture viewport: avoid left/right crop.
        aspect = min(aspect_raw, 1.0)
    else:
        aspect = aspect_raw

    vfov = math.radians(fov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)

    fit_w = image_w * max(1.0, fit_padding)
    fit_h = image_h * max(1.0, fit_padding)

    dist_y = (fit_h * 0.5) / max(1e-6, math.tan(vfov / 2.0))
    dist_x = (fit_w * 0.5) / max(1e-6, math.tan(hfov / 2.0))

    if mode == "longest-axis":
        dist_base = dist_x if fit_w >= fit_h else dist_y
    else:
        # "contain" mode keeps both axes fully visible.
        dist_base = max(dist_x, dist_y)

    dist = dist_base * margin * max(1e-6, distance_scale) * max(1.0, safe_epsilon)

    return (cx, cy, dist), dist


def compute_aligned_frame(
    manifest: dict,
    placement: dict[str, dict[str, float]],
    adjustments_map: dict,
    alignment_samples: dict,
) -> tuple[float, float, float, float]:
    """Return (center_x, center_y, width, height) from aligned mesh bounds."""
    cfg = manifest.get("config", {})
    stats = manifest.get("stats", {})
    fallback_w = float(stats.get("input_width", 1)) * float(cfg.get("pixel_scale", 1.0))
    fallback_h = float(stats.get("input_height", 1)) * float(cfg.get("pixel_scale", 1.0))

    mesh_files: list[str] = manifest.get("mesh_files", [])
    if not mesh_files:
        return fallback_w * 0.5, fallback_h * 0.5, fallback_w, fallback_h

    global_adj = adjustments_map.get("global", {})
    per_mesh_adj = adjustments_map.get("per_mesh", {})
    dx_per_index = float(global_adj.get("dx_per_index", 0.0))
    dy_per_index = float(global_adj.get("dy_per_index", 0.0))
    dz_per_index = float(global_adj.get("dz_per_index", 0.0))

    base_positions: dict[str, tuple[float, float, float]] = {}
    for idx, mesh_file in enumerate(mesh_files):
        place = placement.get(mesh_file, {"cx": 0.0, "cy": 0.0, "cz": 0.0})
        mesh_adj = per_mesh_adj.get(mesh_file, {})
        cx = place["cx"] + (dx_per_index * idx) + float(mesh_adj.get("dx", 0.0))
        cy = place["cy"] + (dy_per_index * idx) + float(mesh_adj.get("dy", 0.0))
        cz = place["cz"] + (dz_per_index * idx) + float(mesh_adj.get("dz", 0.0))
        base_positions[mesh_file] = (cx, cy, cz)

    mode = str(alignment_samples.get("mode", "index-interpolate")).strip().lower()
    if mode == "size-linear":
        auto_deltas = build_size_linear_alignment(mesh_files, placement, base_positions, alignment_samples)
    else:
        auto_deltas = build_interpolated_alignment(mesh_files, base_positions, alignment_samples)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for mesh_file in mesh_files:
        place = placement.get(mesh_file, {"sx": 1.0, "sy": 1.0})
        cx, cy, _ = base_positions.get(mesh_file, (0.0, 0.0, 0.0))
        adx, ady, _ = auto_deltas.get(mesh_file, (0.0, 0.0, 0.0))
        cx += adx
        cy += ady

        sx = float(place.get("sx", 1.0))
        sy = float(place.get("sy", 1.0))
        min_x = min(min_x, cx - (sx * 0.5))
        max_x = max(max_x, cx + (sx * 0.5))
        min_y = min(min_y, cy - (sy * 0.5))
        max_y = max(max_y, cy + (sy * 0.5))

    if not (math.isfinite(min_x) and math.isfinite(max_x) and math.isfinite(min_y) and math.isfinite(max_y)):
        return fallback_w * 0.5, fallback_h * 0.5, fallback_w, fallback_h

    width = max(0.001, max_x - min_x)
    height = max(0.001, max_y - min_y)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    return center_x, center_y, width, height


def _build_generated_model_item(
    manifest: dict,
    asset_map: dict,
    adjustments_map: dict,
    alignment_samples: dict,
    obf_template: dict,
    include_obf: bool,
    model_name: str,
    fov_deg: float,
    margin: float,
    viewport_aspect: float,
    camera_fit_padding: float,
    camera_distance_scale: float,
    camera_safe_epsilon: float,
    camera_fit_mode: str,
    apply_bounds_size: bool,
) -> ET.Element:
    cfg = manifest.get("config", {})
    stats = manifest.get("stats", {})

    image_w = float(stats.get("input_width", 1)) * float(cfg.get("pixel_scale", 1.0))
    image_h = float(stats.get("input_height", 1)) * float(cfg.get("pixel_scale", 1.0))

    model = ET.Element("Item", {"class": "Model", "referent": _ref()})
    model_props = _props(model)
    _add_string(model_props, "Name", model_name)

    # Container folder for generated mesh parts.
    meshes = _add_item(model, "Folder")
    meshes_props = _props(meshes)
    _add_string(meshes_props, "Name", "Meshes")

    mesh_ids = asset_map.get("mesh_ids", {})
    tex_ids = asset_map.get("texture_ids", {})
    placement = build_mesh_placement_map(manifest)
    global_adj = adjustments_map.get("global", {})
    per_mesh_adj = adjustments_map.get("per_mesh", {})
    dx_per_index = float(global_adj.get("dx_per_index", 0.0))
    dy_per_index = float(global_adj.get("dy_per_index", 0.0))
    dz_per_index = float(global_adj.get("dz_per_index", 0.0))

    mesh_files: list[str] = manifest.get("mesh_files", [])
    atlas_files: list[str] = manifest.get("atlas_files", [])

    base_positions: dict[str, tuple[float, float, float]] = {}
    for idx, mesh_file in enumerate(mesh_files):
        place = placement.get(mesh_file, {"cx": 0.0, "cy": 0.0, "cz": 0.0})
        mesh_adj = per_mesh_adj.get(mesh_file, {})
        cx = place["cx"] + (dx_per_index * idx) + float(mesh_adj.get("dx", 0.0))
        cy = place["cy"] + (dy_per_index * idx) + float(mesh_adj.get("dy", 0.0))
        cz = place["cz"] + (dz_per_index * idx) + float(mesh_adj.get("dz", 0.0))
        base_positions[mesh_file] = (cx, cy, cz)

    mode = str(alignment_samples.get("mode", "index-interpolate")).strip().lower()
    if mode == "size-linear":
        auto_deltas = build_size_linear_alignment(mesh_files, placement, base_positions, alignment_samples)
    else:
        auto_deltas = build_interpolated_alignment(mesh_files, base_positions, alignment_samples)

    for idx, mesh_file in enumerate(mesh_files):
        page = parse_page_from_mesh_name(mesh_file)
        atlas_file = atlas_files[page] if 0 <= page < len(atlas_files) else atlas_files[0]

        mesh_id = normalize_asset_id(mesh_ids.get(mesh_file))
        tex_id = normalize_asset_id(tex_ids.get(atlas_file))
        place = placement.get(mesh_file, {"cx": 0.0, "cy": 0.0, "cz": 0.0, "sx": 1.0, "sy": 1.0, "sz": 0.05})
        cx, cy, cz = base_positions.get(mesh_file, (0.0, 0.0, 0.0))
        adx, ady, adz = auto_deltas.get(mesh_file, (0.0, 0.0, 0.0))
        cx += adx
        cy += ady
        cz += adz

        part = _add_item(meshes, "MeshPart")
        p = _props(part)
        _add_string(p, "Name", Path(mesh_file).stem)
        _add_bool(p, "Anchored", False)
        _add_bool(p, "CanCollide", False)
        _add_bool(p, "CanQuery", False)
        _add_bool(p, "CanTouch", False)
        _add_bool(p, "DoubleSided", True)
        _add_float(p, "Transparency", 0.0)
        _add_token(p, "Material", 256)  # Plastic
        _set_smooth_surfaces(p)
        _add_cframe_identity(p, "CFrame", cx, cy, cz)
        if apply_bounds_size:
            _add_vector3(p, "Size", place["sx"], place["sy"], place["sz"])
        else:
            # Keep identity size to avoid importer-dependent stretch artifacts.
            _add_vector3(p, "Size", 1.0, 1.0, 1.0)
        _add_content(p, "MeshId", mesh_id)
        _add_content(p, "TextureID", tex_id)

    frame_cx, frame_cy, _, _ = compute_aligned_frame(
        manifest=manifest,
        placement=placement,
        adjustments_map=adjustments_map,
        alignment_samples=alignment_samples,
    )

    # Geometric camera fit from scratch:
    # 1) center from aligned mesh layout
    # 2) size from exact canvas dimensions
    fit_w = image_w
    fit_h = image_h
    cam_pos, cam_dist = compute_camera(
        fit_w,
        fit_h,
        fov_deg=fov_deg,
        margin=margin,
        viewport_aspect=viewport_aspect,
        fit_padding=camera_fit_padding,
        distance_scale=camera_distance_scale,
        safe_epsilon=camera_safe_epsilon,
        fit_mode=camera_fit_mode,
        center_x=frame_cx,
        center_y=frame_cy,
    )
    cam_pos = (cam_pos[0], cam_pos[1], -cam_dist)
    cam_focus = (frame_cx, frame_cy, 0.0)

    if include_obf:
        add_obf_models(
            parent_model=model,
            image_w=image_w,
            image_h=image_h,
            obf_template=obf_template,
            z_center=0.0,
            z_span=0.12,
        )

    # Lighting helper: centered spotlight aimed toward the mesh plane.
    light_part = _add_item(model, "Part")
    lp = _props(light_part)
    _add_string(lp, "Name", "CenterSpotLightPart")
    _add_bool(lp, "Anchored", False)
    _add_bool(lp, "CanCollide", False)
    _add_bool(lp, "CanQuery", False)
    _add_bool(lp, "CanTouch", False)
    _add_bool(lp, "Locked", True)
    _add_float(lp, "Transparency", 1.0)
    _add_token(lp, "Material", 256)
    _set_smooth_surfaces(lp)
    _add_vector3(lp, "Size", 1.0, 1.0, 1.0)
    _add_cframe_look_at(lp, "CFrame", (frame_cx, frame_cy, -5.3), (frame_cx, frame_cy, 0.0))

    spot = _add_item(light_part, "SpotLight")
    sp = _props(spot)
    _add_string(sp, "Name", "CenterSpotLight")
    _add_float(sp, "Angle", 180.0)
    _add_float(sp, "Brightness", 0.78)
    _add_color3(sp, "Color", 255, 255, 255)
    _add_float(sp, "Range", 60.0)
    _add_bool(sp, "Shadows", False)
    _add_token(sp, "Face", 5)

    # Camera object for quick framing reference.
    cam = _add_item(model, "Camera")
    cam_props = _props(cam)
    _add_string(cam_props, "Name", "ThumbnailCamera")
    _add_float(cam_props, "FieldOfView", fov_deg)
    _add_cframe_look_at(cam_props, "CFrame", cam_pos, cam_focus)
    _add_cframe_identity(cam_props, "Focus", cam_focus[0], cam_focus[1], cam_focus[2])

    # Metadata values for tools/scripts to apply the camera automatically.
    cfg_item = _add_item(model, "Configuration")
    cfg_props = _props(cfg_item)
    _add_string(cfg_props, "Name", "ExportConfig")

    cam_cframe_value = _add_item(cfg_item, "CFrameValue")
    cam_cframe_props = _props(cam_cframe_value)
    _add_string(cam_cframe_props, "Name", "RecommendedCameraCFrame")
    _add_cframe_look_at(cam_cframe_props, "Value", cam_pos, cam_focus)

    cam_focus_value = _add_item(cfg_item, "CFrameValue")
    cam_focus_props = _props(cam_focus_value)
    _add_string(cam_focus_props, "Name", "RecommendedCameraFocus")
    _add_cframe_identity(cam_focus_props, "Value", cam_focus[0], cam_focus[1], cam_focus[2])

    fov_value = _add_item(cfg_item, "NumberValue")
    fov_props = _props(fov_value)
    _add_string(fov_props, "Name", "RecommendedFieldOfView")
    _add_float(fov_props, "Value", fov_deg)

    return model


def _new_roblox_root() -> ET.Element:
    return ET.Element("roblox", {
        "xmlns:xmime": "http://www.w3.org/2005/05/xmlmime",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://www.roblox.com/roblox.xsd",
        "version": "4",
    })


def build_rbxmx(
    manifest: dict,
    asset_map: dict,
    adjustments_map: dict,
    alignment_samples: dict,
    obf_template: dict,
    include_obf: bool,
    model_name: str,
    fov_deg: float,
    margin: float,
    viewport_aspect: float,
    camera_fit_padding: float,
    camera_distance_scale: float,
    camera_safe_epsilon: float,
    camera_fit_mode: str,
    apply_bounds_size: bool,
) -> ET.ElementTree:
    root = _new_roblox_root()
    root.append(
        _build_generated_model_item(
            manifest,
            asset_map,
            adjustments_map,
            alignment_samples,
            obf_template,
            include_obf,
            model_name,
            fov_deg,
            margin,
            viewport_aspect,
            camera_fit_padding,
            camera_distance_scale,
            camera_safe_epsilon,
            camera_fit_mode,
            apply_bounds_size,
        )
    )
    return ET.ElementTree(root)


def build_rbxlx(
    manifest: dict,
    asset_map: dict,
    adjustments_map: dict,
    alignment_samples: dict,
    obf_template: dict,
    include_obf: bool,
    model_name: str,
    fov_deg: float,
    margin: float,
    viewport_aspect: float,
    camera_fit_padding: float,
    camera_distance_scale: float,
    camera_safe_epsilon: float,
    camera_fit_mode: str,
    apply_bounds_size: bool,
) -> ET.ElementTree:
    cfg = manifest.get("config", {})
    stats = manifest.get("stats", {})

    image_w = float(stats.get("input_width", 1)) * float(cfg.get("pixel_scale", 1.0))
    image_h = float(stats.get("input_height", 1)) * float(cfg.get("pixel_scale", 1.0))
    placement = build_mesh_placement_map(manifest)
    frame_cx, frame_cy, _, _ = compute_aligned_frame(
        manifest=manifest,
        placement=placement,
        adjustments_map=adjustments_map,
        alignment_samples=alignment_samples,
    )

    # Geometric camera fit from scratch:
    # 1) center from aligned mesh layout
    # 2) size from exact canvas dimensions
    fit_w = image_w
    fit_h = image_h
    cam_pos, cam_dist = compute_camera(
        fit_w,
        fit_h,
        fov_deg=fov_deg,
        margin=margin,
        viewport_aspect=viewport_aspect,
        fit_padding=camera_fit_padding,
        distance_scale=camera_distance_scale,
        safe_epsilon=camera_safe_epsilon,
        fit_mode=camera_fit_mode,
        center_x=frame_cx,
        center_y=frame_cy,
    )
    cam_pos = (cam_pos[0], cam_pos[1], -cam_dist)
    cam_focus = (frame_cx, frame_cy, 0.0)

    root = _new_roblox_root()
    workspace = _add_item(root, "Workspace")
    ws_props = _props(workspace)
    _add_string(ws_props, "Name", "Workspace")

    world_cam = _add_item(workspace, "Camera")
    cam_ref = world_cam.get("referent", "")
    world_cam_props = _props(world_cam)
    _add_string(world_cam_props, "Name", "SceneCamera")
    _add_float(world_cam_props, "FieldOfView", fov_deg)
    _add_cframe_look_at(world_cam_props, "CFrame", cam_pos, cam_focus)
    _add_cframe_identity(world_cam_props, "Focus", cam_focus[0], cam_focus[1], cam_focus[2])

    # Set Workspace.CurrentCamera to the generated scene camera for instant framing.
    ws_current_camera = ET.SubElement(ws_props, "Ref", {"name": "CurrentCamera"})
    ws_current_camera.text = cam_ref

    model_item = _build_generated_model_item(
        manifest,
        asset_map,
        adjustments_map,
        alignment_samples,
        obf_template,
        include_obf,
        model_name,
        fov_deg,
        margin,
        viewport_aspect,
        camera_fit_padding,
        camera_distance_scale,
        camera_safe_epsilon,
        camera_fit_mode,
        apply_bounds_size,
    )
    workspace.append(copy.deepcopy(model_item))

    return ET.ElementTree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Roblox .rbxmx model from patch_mesh_manifest.json and asset IDs")
    parser.add_argument("--manifest", type=Path, default=MESH_RESULTS_DIR / "patch_mesh_manifest.json")
    parser.add_argument("--asset-map", type=Path, default=SCRIPT_DIR / "roblox_asset_ids.json")
    parser.add_argument("--adjustments", type=Path, default=SCRIPT_DIR / "roblox_mesh_adjustments.json")
    parser.add_argument("--alignment-samples", type=Path, default=SCRIPT_DIR / "roblox_alignment_samples.json")
    parser.add_argument("--obf-template", type=Path, default=SCRIPT_DIR / "roblox_obf_template.json")
    parser.add_argument("--output-dir", type=Path, default=ROBLOX_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rbxlx-output", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="GeneratedMeshModel")
    parser.add_argument("--camera-fov", type=float, default=35.0)
    parser.add_argument("--camera-margin", type=float, default=1.0)
    parser.add_argument(
        "--camera-aspect",
        type=float,
        default=1.0,
        help="Assumed viewport width/height for camera fit. Increase safety for extreme ratios.",
    )
    parser.add_argument(
        "--camera-fit-padding",
        type=float,
        default=1.0,
        help="Extra image bounds padding used by camera fit to avoid edge clipping.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.0,
        help="Final camera distance multiplier (<1 zooms in, >1 zooms out).",
    )
    parser.add_argument(
        "--camera-safe-epsilon",
        type=float,
        default=1.0005,
        help="Tiny safety multiplier to avoid rounding clip while keeping near edge-fit framing.",
    )
    parser.add_argument(
        "--camera-fit-mode",
        choices=["longest-axis", "contain"],
        default="contain",
        help="Camera fit strategy: prioritize longer image axis or contain both axes.",
    )
    parser.add_argument(
        "--apply-bounds-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale MeshPart.Size from OBJ bounds. Disable to avoid stretch from importer normalization.",
    )
    parser.add_argument(
        "--include-obf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include reusable obf model cloud (9 models x 546 transparent parts).",
    )
    parser.add_argument("--obf-model-count", type=int, default=DEFAULT_OBF_MODELS)
    parser.add_argument("--obf-parts-per-model", type=int, default=DEFAULT_OBF_PARTS_PER_MODEL)
    parser.add_argument("--obf-seed", type=int, default=734921)
    return parser.parse_args()


def prepare_clean_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            # Skip files that are temporarily locked by external tools.
            print(f"Skipping locked file during cleanup: {child}")
    return output_dir


def main() -> None:
    args = parse_args()

    if args.camera_fov <= 1.0 or args.camera_fov >= 120.0:
        raise ValueError("camera-fov must be between 1 and 120 degrees")
    if args.camera_margin <= 0:
        raise ValueError("camera-margin must be > 0")
    if args.camera_aspect <= 0:
        raise ValueError("camera-aspect must be > 0")
    if args.camera_fit_padding < 1.0:
        raise ValueError("camera-fit-padding must be >= 1.0")
    if args.camera_distance_scale <= 0:
        raise ValueError("camera-distance-scale must be > 0")
    if args.camera_safe_epsilon < 1.0:
        raise ValueError("camera-safe-epsilon must be >= 1.0")
    if args.obf_model_count <= 0:
        raise ValueError("obf-model-count must be > 0")
    if args.obf_parts_per_model <= 0:
        raise ValueError("obf-parts-per-model must be > 0")

    output_dir = prepare_clean_output_dir(args.output_dir)
    output_path = Path(args.output) if args.output is not None else (output_dir / "generated_model.rbxmx")
    rbxlx_output_path = (
        Path(args.rbxlx_output) if args.rbxlx_output is not None else (output_dir / "generated_place.rbxlx")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rbxlx_output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    asset_map = ensure_asset_map(args.asset_map, manifest)
    adjustments_map = ensure_adjustments_map(args.adjustments, manifest)
    alignment_samples = ensure_alignment_samples(args.alignment_samples, manifest)
    obf_template = ensure_obf_template(
        args.obf_template,
        model_count=args.obf_model_count,
        parts_per_model=args.obf_parts_per_model,
        seed=args.obf_seed,
    )

    tree = build_rbxmx(
        manifest=manifest,
        asset_map=asset_map,
        adjustments_map=adjustments_map,
        alignment_samples=alignment_samples,
        obf_template=obf_template,
        include_obf=args.include_obf,
        model_name=args.model_name,
        fov_deg=args.camera_fov,
        margin=args.camera_margin,
        viewport_aspect=args.camera_aspect,
        camera_fit_padding=args.camera_fit_padding,
        camera_distance_scale=args.camera_distance_scale,
        camera_safe_epsilon=args.camera_safe_epsilon,
        camera_fit_mode=args.camera_fit_mode,
        apply_bounds_size=args.apply_bounds_size,
    )

    place_tree = build_rbxlx(
        manifest=manifest,
        asset_map=asset_map,
        adjustments_map=adjustments_map,
        alignment_samples=alignment_samples,
        obf_template=obf_template,
        include_obf=args.include_obf,
        model_name=args.model_name,
        fov_deg=args.camera_fov,
        margin=args.camera_margin,
        viewport_aspect=args.camera_aspect,
        camera_fit_padding=args.camera_fit_padding,
        camera_distance_scale=args.camera_distance_scale,
        camera_safe_epsilon=args.camera_safe_epsilon,
        camera_fit_mode=args.camera_fit_mode,
        apply_bounds_size=args.apply_bounds_size,
    )

    ET.indent(tree, space="  ")
    ET.indent(place_tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    place_tree.write(rbxlx_output_path, encoding="utf-8", xml_declaration=True)

    print(f"Wrote: {output_path}")
    print(f"Wrote: {rbxlx_output_path}")
    print(f"Asset ID map: {args.asset_map}")
    print(f"Adjustment map: {args.adjustments}")
    print(f"Alignment samples: {args.alignment_samples}")
    print(f"Obf template: {args.obf_template}")
    print("If you still have 0 placeholders in asset IDs, fill them and run again.")


if __name__ == "__main__":
    main()
