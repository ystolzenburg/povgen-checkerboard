import subprocess
import os
from PIL import Image
import numpy as np
import sys
import argparse

script_background = "background { color rgb <0.27, 0.27, 0.27> }"
script_head = """
#declare lens=camera{perspective location <0, 16,-50>  look_at <0,0,0>  angle 12};
camera{lens}
light_source{<20, 10, 7>  color rgb <1.00, 1.00, 1.00> area_light 6*x, 6*y, 12, 12}
union{"""

script_end_x = """
rotate y * 45}
polygon{4, <-2.000000, 1.850000, -8.000000> <-2.000000, -0.200000, -8.000000> <2.000000, -0.200000, -8.000000> <2.000000, 1.850000, -8.000000>
rotate x *15
translate<0, -2.3, 0>
texture{pigment {color rgb <2.126000, 2.126000, 2.126000> transmit 0.400000}}}
"""
script_end = "rotate y * 45}"

reference_end = """
rotate y * 45}
polygon{4, <-2.000000, 1.850000, -8.000000> <-2.000000, -0.200000, -8.000000> <2.000000, -0.200000, -8.000000> <2.000000, 1.850000, -8.000000>
rotate x *15
translate<0, -2.3, 0>
texture{pigment {color rgb <1,1,1>}}}
"""

def michelson_contrast(L1, L2, eps=1e-6):
    return abs(L1 - L2) / (L1 + L2 + eps)

def get_neighbors(array: np.ndarray, i: int, j: int):
    neighbors = []
    rows, cols = array.shape
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbors.append(array[ni, nj])
    return neighbors

def gen_shades(
    grid_size: int,
    target_luminance: float = 0.5,
    diff_interval: list[float] = [0.05, 1],
    max_attempts: int = 1_000_000_000,
    accuracy: float = 0.001
) -> np.ndarray:
    grid = np.full((grid_size, grid_size), np.random.uniform(0, 1))
    luminance_attempts = 0
    while (abs(grid.mean() - target_luminance) > accuracy and luminance_attempts <= max_attempts):
        for i in range(grid_size):
            for j in range(grid_size):
                attempts = 0
                while attempts < max_attempts:
                    print(f"\rAttempt {luminance_attempts} to match luminance within reach of {accuracy}. Currently {grid.mean()}. Attempt {attempts} trying to find matching colors for {[i,j]}", end="", flush=True)
                    random_color = np.random.uniform(0, 1)
                    nbs = get_neighbors(grid, i, j)
                    valid = True
                    for k in nbs:
                        diff = abs(random_color - k)
                        lower, upper = min(diff_interval), max(diff_interval)
                        if not (lower <= diff <= upper):
                            valid = False
                            break
                    if valid:
                        grid[i][j] = random_color
                        break
                    attempts += 1
    
                if grid[i][j] == -1:
                    print(f"Couldn't generate valid color at ({i},{j}) after {attempts} attempts.")
                    exit(1)
        luminance_attempts += 1    
    print(f"\nGrid Luminance: {grid.mean()}", end="\n", flush=True)
    return grid



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


def gen_grid(grid_size: int, box_size: float, colors: np.ndarray):
    base = -(box_size*grid_size)/2

    return_str = ""
    for i in range(grid_size):
        for j in range(grid_size):
            color = colors[i,j]
            return_str += add_box(
                Coordinate3D(base+box_size*j,     -1,   base+box_size*i),
                Coordinate3D(base+box_size*(j+1), -0.71,base+box_size*(i+1)),
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
    parser.add_argument("--grid-size", type=int, help="set checkerboard grid size", default=8)
    parser.add_argument("--cube-size", type=float, help="set single checkerbox cube size", default=0.73)
    parser.add_argument("allowed_differences", nargs=4, type=float, help="<min bg> <max bg> <min cutout> <max cutout>")
    args = parser.parse_args()
    min_diff_bg, max_diff_bg, min_diff_ct ,max_diff_ct = args.allowed_differences
    cube_size = args.cube_size

    print("calculating background colors (difference min:", str(min_diff_bg), "max:", str(max_diff_bg),")...")
    colors_bg = gen_shades(grid_size=args.grid_size, diff_interval=[min_diff_bg, max_diff_bg])
    
    print("rendering background with " + str(args.grid_size) + "x" + str(args.grid_size) + " cubes...")
    render_string_bg = script_background+script_head+gen_grid(args.grid_size, cube_size, colors_bg)+script_end
    background = render_img(render_string_bg, "background")
    
    print("calculating cutout colors (difference min:", str(min_diff_ct), "max:", str(max_diff_ct),")...")
    colors_ct = gen_shades(grid_size=args.grid_size, diff_interval=[min_diff_ct, max_diff_ct])
        
    print("rendering cutout with " + str(args.grid_size) + "x" + str(args.grid_size) + " cubes...")
    render_string_ct = script_background+script_head+gen_grid(args.grid_size, cube_size, colors_ct)+script_end
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
        open_img("result.png")
    else:
        print("crop box couldn't be generated")

if __name__ == "__main__":
    main()