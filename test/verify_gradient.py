import numpy as np
import trimesh
import argparse
import torch
import cubvh
import time


def create_dodecahedron(radius=1, center=np.array([0, 0, 0])):
    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    faces = np.array([
        19, 3, 2,
        12, 19, 2,
        15, 12, 2,
        8, 14, 2,
        18, 8, 2,
        3, 18, 2,
        20, 5, 4,
        9, 20, 4,
        16, 9, 4,
        13, 17, 4,
        1, 13, 4,
        5, 1, 4,
        7, 16, 4,
        6, 7, 4,
        17, 6, 4,
        6, 15, 2,
        7, 6, 2,
        14, 7, 2,
        10, 18, 3,
        11, 10, 3,
        19, 11, 3,
        11, 1, 5,
        10, 11, 5,
        20, 10, 5,
        20, 9, 8,
        10, 20, 8,
        18, 10, 8,
        9, 16, 7,
        8, 9, 7,
        14, 8, 7,
        12, 15, 6,
        13, 12, 6,
        17, 13, 6,
        13, 1, 11,
        12, 13, 11,
        19, 12, 11,
        ]).reshape((-1, 3), order="C")-1

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    return trimesh.Trimesh(vertices=vertices, faces=faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--mesh', default='', type=str)
    
    opt = parser.parse_args()

    if opt.mesh == '':
        mesh = create_dodecahedron()
    else:
        mesh = trimesh.load(opt.mesh, force='mesh', skip_material=True)


    # query nearest triangles for a set of points
    points = torch.randn(opt.N, 3, device='cuda', dtype=torch.float32)

    # Ours
    _t0 = time.time()
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
    torch.cuda.synchronize()
    _t1 = time.time()
    
    # test grad
    points.requires_grad_(True)
    distances, _, _ = BVH.unsigned_distance(points)
    # loss = (distances ** 2).sum()
    loss = distances.sum()
    loss.backward()

    grad = points.grad.cpu().numpy()
    
    # numerical gradient
    eps = 1e-4
    grad_num = np.zeros_like(grad)
    for i in range(3):
        points_pos = points.detach().clone()
        points_pos[:, i] += eps
        dist_pos, _, _ = BVH.unsigned_distance(points_pos)

        points_neg = points.detach().clone()
        points_neg[:, i] -= eps
        dist_neg, _, _ = BVH.unsigned_distance(points_neg)

        # grad_num[:, i] = ((dist_pos ** 2).cpu().numpy() - (dist_neg ** 2).cpu().numpy()) / (2 * eps)
        grad_num[:, i] = (dist_pos.cpu().numpy() - dist_neg.cpu().numpy()) / (2 * eps)

    # verify correctness
    np.testing.assert_allclose(
        grad,
        grad_num,
        atol=1e-5
    )
    
    print(f'[INFO] gradient verification passed!')
