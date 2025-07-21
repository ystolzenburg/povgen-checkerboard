import subprocess
import os
from PIL import Image
import numpy as np
import sys
import argparse

script_background = "background { color rgb <0.27, 0.27, 0.27> }"
script_head = """
#version 3.7
#declare lens=camera{perspective location <0, 16,-50>  look_at <0,0,0>  angle 12};
camera{lens}
light_source{<20, 10, 7>  color rgb <1.00, 1.00, 1.00> area_light 6*x, 6*y, 12, 12}
union{"""

script_end = """
rotate y * 45}
polygon{4, <-2.000000, 1.850000, -8.000000> <-2.000000, -0.200000, -8.000000> <2.000000, -0.200000, -8.000000> <2.000000, 1.850000, -8.000000>
rotate x *15
translate<0, -2.3, 0>
texture{pigment {color rgb <2.126000, 2.126000, 2.126000> transmit 0.400000}}}
"""
script_end_x = "rotate y * 45}"

reference_end = """
rotate y * 45}
polygon{4, <-2.000000, 1.850000, -8.000000> <-2.000000, -0.200000, -8.000000> <2.000000, -0.200000, -8.000000> <2.000000, 1.850000, -8.000000>
rotate x *15
translate<0, -2.3, 0>
texture{pigment {color rgb <1,1,1>}}}
"""

def get_neighbors(grid, i, j):
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 8 and 0 <= nj < 8 and grid[ni][nj] >= 0:
            neighbors.append(grid[ni][nj])
    return neighbors

def generate_luminance_matched_values(target_luminance, total=64):
    vals = np.random.uniform(0, 1, total)
    current_mean = vals.mean()
    diff = target_luminance - current_mean
    vals += diff
    vals = np.clip(vals, 0, 1)
    vals *= target_luminance / vals.mean()
    return vals

def is_valid_placement(grid, i, j, val, diff_interval):
    neighbors = get_neighbors(grid, i, j)
    for nb in neighbors:
        diff = abs(val - nb)
        if not (diff_interval[0] <= diff and diff <= diff_interval[1]):
            return False
    return True

def fill_grid(values, diff_interval):
    grid = np.full((8, 8), -1.0)
    positions = [(i, j) for i in range(8) for j in range(8)]
    idx = 0

    for i, j in positions:
        placed = False
        for _ in range(len(values)):
            val = values[idx % len(values)]
            idx += 1
            if is_valid_placement(grid, i, j, val, diff_interval):
                grid[i, j] = val
                placed = True
                break
        if not placed:
            return None
    return grid

def gen_shades(target_luminance=0.5, diff_interval=[0.05, 1.0], attempts=100_000, accuracy=0.00001):
    for _ in range(attempts):
        values = generate_luminance_matched_values(target_luminance)
        np.random.shuffle(values)
        grid = fill_grid(values, diff_interval)
        if grid is not None and abs(grid.mean() - target_luminance) <= accuracy:
            return grid
    raise ValueError("Failed to generate a valid grid within constraints")


class Coordinate3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "<"+f"{self.x:.6f}"+", "+f"{self.y:.6f}"+", "+f"{self.z:.6f}"+">"

    def __eq__(self, other):
            if not isinstance(other, Coordinate3D):
                return False
            return abs(self.x - other.x) < 0.1 and abs(self.y - other.y) < 0.1 and abs(self.z - other.z) < 0.1


def add_box(topLeft: Coordinate3D, bottomRight: Coordinate3D, color: Coordinate3D) -> str:
    return "box{"+str(topLeft)+","+str(bottomRight)+" pigment{ color rgb "+str(color)+" }}\n"


def gen_grid(colors: np.ndarray):
    base = -(0.73*8)/2
    return_str = ""
    for i in range(8):
        for j in range(8):
            color = colors[i,j]
            return_str += add_box(
                Coordinate3D(base+0.73*j,     -1,   base+0.73*i),
                Coordinate3D(base+0.73*(j+1), -0.71,base+0.73*(i+1)),
                Coordinate3D(color, color, color))

    return return_str

def get_crop_box(image_path):
    img = Image.open(image_path).convert("L")
    pixels = img.load()

    width, height = img.size
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for y in range(height):
        for x in range(width):
            if pixels != None and pixels[x, y] > 10:  # anything not black
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

    if min_x > max_x or min_y > max_y:
        return None

    return (min_x, min_y, max_x + 1, max_y + 1)  # (left, upper, right, lower)

def render_img(script: str, output_name: str) -> str:
    os.makedirs("images", exist_ok=True)
    pov_file_path = os.path.join("scene.pov")
    
    with open(pov_file_path, "w", encoding="utf-8") as f:
        f.write(script)

    output_file = os.path.join("images", output_name+".png")

    try:
        subprocess.run([
            "povray",
            f"+I{pov_file_path}",
            f"+O{output_file}",
            "+W1200", "+H1200", "+A0.1", "+FN", "-GA"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_file
    finally:
        os.remove(pov_file_path)
        
def open_img(path):
    imageViewerFromCommandLine = {'linux':'xdg-open',
                                  'win32':'explorer',
                                  'darwin':'open'}[sys.platform]
    subprocess.Popen([imageViewerFromCommandLine, path])

def main():
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("allowed_differences", nargs=3, type=float, help="<min> <max> <alpha>")
    args = parser.parse_args()
    min, max, alpha = args.allowed_differences

    print("calculating background colors (difference min:", str(min), "max:", str(max),")...")
    shades = gen_shades(diff_interval=[min, max])
    m = shades.mean()
    compressed_shades = (shades - m) * alpha + m    
    
    print("Cutout mean:", m, "New mean:", compressed_shades.mean())
    print("Cutout range:", shades.min(), "to", shades.max())
    print("Compressed range:", compressed_shades.min(), "to", compressed_shades.max())
    print("rendering background with 8x8 cubes...")
    render_string_bg = script_background+script_head+gen_grid(compressed_shades)+script_end
    background = render_img(render_string_bg, "background")
        
    print("rendering cutout with 8x8 cubes...")
    render_string_ct = script_background+script_head+gen_grid(shades)+script_end
    cutout = render_img(render_string_ct, "cutout")
    
    print("rendering cropping range...")
    render_string_crop = "background { color rgb <0, 0, 0> }"+script_head+reference_end
    crop_reference = render_img(render_string_crop, "crop_reference")
    crop_box = get_crop_box(crop_reference)
    
    if crop_box:
        print("combining images ...")
        source_img = Image.open(cutout).convert("RGBA")
        target_img = Image.open(background).convert("RGBA")
        
        cropped = source_img.crop(crop_box)
        target_img.paste(cropped, crop_box[:2], cropped)
        target_img.save("result.png")
        print("done!")
        open_img("images/cutout.png")
        open_img("result.png")
    else:
        print("crop box couldn't be generated")

if __name__ == "__main__":
    main()