import os
import glob
import tqdm
import time
import trimesh
import argparse
import numpy as np

import torch
from cubvh.sparcubes import Sparcubes

"""
Extract watertight mesh from an arbitrary mesh using the Sparcubes class.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--res', type=int, default=256) # Lowered default for faster testing
parser.add_argument('--workspace', type=str, default='output')
parser.add_argument('--deformed_grid', action='store_true', help="Use sparse marching cubes on deformed grid.")
opt = parser.parse_args()

device = torch.device('cuda')

def run(path):
    print(f"[INFO] Processing {path}")
    try:
        mesh = trimesh.load(path, process=False, force='mesh')
        
        vertices = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)

        t0 = time.time()
        sparcube_generator = Sparcubes(vertices, faces, resolution=opt.res, device=device)
        print(f'Sparcubes setup time: {time.time() - t0:.4f}s')

        t0 = time.time()
        watertight_mesh = sparcube_generator.extract_mesh(level=0, use_deformed_grid=opt.deformed_grid)
        print(f'Mesh extraction time: {time.time() - t0:.4f}s, vertices: {len(watertight_mesh.vertices)}, triangles: {len(watertight_mesh.faces)}')

        name = os.path.splitext(os.path.basename(path))[0]
        output_path = f'{opt.workspace}/{name}_watertight.obj'
        watertight_mesh.export(output_path)
        print(f"[INFO] Saved watertight mesh to {output_path}")

    except Exception as e:
        print(f'[ERROR] Failed to process {path}: {e}')
        import traceback
        traceback.print_exc()


os.makedirs(opt.workspace, exist_ok=True)

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    for path in tqdm.tqdm(file_paths):
        run(path)
else:
    run(opt.test_path)
