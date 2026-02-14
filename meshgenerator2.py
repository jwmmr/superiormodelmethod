from PIL import Image
import numpy as np
from collections import defaultdict
import math
import os

def generate_color_atlas(input_path, output_path="color_atlas.png"):
    """Generate a 1024x1024 texture with all unique colors including partial transparency"""
    orig_img = Image.open(input_path).convert('RGBA')  # Force RGBA for alpha handling
    orig_pixels = np.array(orig_img)
    h, w, _ = orig_pixels.shape
    
    # Collect all unique colors with some visibility (alpha > 0)
    unique_colors = set()
    for y in range(h):
        for x in range(w):
            color = tuple(orig_pixels[y, x])
            if color[3] > 0:  # Only include colors with some transparency
                unique_colors.add(color)
    
    unique_colors = list(unique_colors)
    num_colors = len(unique_colors)
    print(f"Found {num_colors} colors with visibility (alpha > 0)")

    # Target texture size
    tex_size = 1024
    max_colors = tex_size * tex_size
    
    if num_colors > max_colors:
        print(f"Using first {max_colors} of {num_colors} colors")
        unique_colors = unique_colors[:max_colors]
        num_colors = max_colors
    
    # Calculate optimal grid dimensions
    cols = math.ceil(math.sqrt(num_colors))
    rows = math.ceil(num_colors / cols)
    
    # Create texture with alpha channel
    texture = np.zeros((tex_size, tex_size, 4), dtype=np.uint8)
    
    # Fill texture and create color mapping
    color_to_uv = {}
    cell_w = tex_size / cols
    cell_h = tex_size / rows
    
    for i, color in enumerate(unique_colors):
        col = i % cols
        row = i // cols
        
        x_start = int(col * cell_w)
        x_end = int((col + 1) * cell_w)
        y_start = int(row * cell_h)
        y_end = int((row + 1) * cell_h)
        
        texture[y_start:y_end, x_start:x_end] = color
        
        # UV coordinates (center of cell)
        u = (x_start + x_end) / (2 * tex_size)
        v = (y_start + y_end) / (2 * tex_size)
        color_to_uv[color] = (u, v)
    
    # Fill remaining space with last color (full opacity)
    for i in range(num_colors, cols * rows):
        col = i % cols
        row = i // cols
        x_start = int(col * cell_w)
        x_end = int((col + 1) * cell_w)
        y_start = int(row * cell_h)
        y_end = int((row + 1) * cell_h)
        
        texture[y_start:y_end, x_start:x_end] = (*unique_colors[-1][:3], 255)
    
    Image.fromarray(texture).save(output_path)
    print(f"Color atlas saved to {output_path}")
    return output_path, color_to_uv, (w, h)

def is_tile_fully_transparent(orig_pixels, x_start, x_end, y_start, y_end):
    """Check if ALL pixels in tile are completely transparent (alpha=0)"""
    tile_alpha = orig_pixels[y_start:y_end, x_start:x_end, 3]
    return np.all(tile_alpha == 0)

def create_tiled_meshes(input_path, color_atlas_path, color_to_uv, original_dims, max_tris=10000):
    """Create mesh tiles, skipping only 100% transparent tiles"""
    orig_w, orig_h = original_dims
    orig_img = Image.open(input_path).convert('RGBA')
    orig_pixels = np.array(orig_img)
    
    # Calculate tiling (2 triangles per pixel)
    max_pixels_per_mesh = max_tris // 2
    cols_per_mesh = min(orig_w, math.isqrt(max_pixels_per_mesh))
    rows_per_mesh = min(orig_h, max_pixels_per_mesh // cols_per_mesh)
    
    num_meshes_x = math.ceil(orig_w / cols_per_mesh)
    num_meshes_y = math.ceil(orig_h / rows_per_mesh)
    print(f"Splitting into {num_meshes_x}x{num_meshes_y} tiles")
    
    mesh_paths = []
    skipped_tiles = 0

    for tile_y in range(num_meshes_y):
        for tile_x in range(num_meshes_x):
            # Calculate tile boundaries
            x_start = tile_x * cols_per_mesh
            x_end = min((tile_x + 1) * cols_per_mesh, orig_w)
            y_start = tile_y * rows_per_mesh
            y_end = min((tile_y + 1) * rows_per_mesh, orig_h)
            
            # Skip only if ENTIRE tile is transparent
            if is_tile_fully_transparent(orig_pixels, x_start, x_end, y_start, y_end):
                skipped_tiles += 1
                continue
            
            tile_name = f"pixel_mesh_{tile_x}_{tile_y}.obj"
            mesh_paths.append(tile_name)
            
            with open(tile_name, 'w') as f:
                f.write(f"# Tile {tile_x},{tile_y}\n")
                
                # Vertex index tracking
                vertex_index = 1
                uv_index = 1
                face_index = 1
                index_map = {}
                
                # Process all pixels in tile
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        color = tuple(orig_pixels[y, x])
                        
                        # Write vertices (flipped Y)
                        f.write(f"v {x} {orig_h-y} 0\n")
                        f.write(f"v {x+1} {orig_h-y} 0\n")
                        f.write(f"v {x+1} {orig_h-y-1} 0\n")
                        f.write(f"v {x} {orig_h-y-1} 0\n")
                        
                        # Write UVs (same for all quad vertices)
                        u, v = color_to_uv.get(color, (0, 0))
                        f.write(f"vt {u} {1-v}\n")
                        f.write(f"vt {u} {1-v}\n")
                        f.write(f"vt {u} {1-v}\n")
                        f.write(f"vt {u} {1-v}\n")
                        
                        # Only write face if pixel has some visibility
                        if color[3] > 0:
                            f.write(f"f {vertex_index}/{uv_index} {vertex_index+1}/{uv_index+1} {vertex_index+2}/{uv_index+2} {vertex_index+3}/{uv_index+3}\n")
                        
                        vertex_index += 4
                        uv_index += 4
    
    print(f"Generated {len(mesh_paths)} tiles (skipped {skipped_tiles} fully transparent tiles)")
    return mesh_paths

if __name__ == "__main__":
    input_image = "input.png"  # Change to your image path
    
    # Step 1: Generate color atlas
    print("Generating color atlas...")
    atlas_path, uv_map, orig_dims = generate_color_atlas(input_image)
    
    # Step 2: Create tiled meshes
    print("Creating mesh tiles...")
    mesh_files = create_tiled_meshes(input_image, atlas_path, uv_map, orig_dims)
    
    print(f"Process complete! Created {len(mesh_files)} mesh files.")