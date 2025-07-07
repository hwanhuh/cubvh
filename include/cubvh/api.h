#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

// abstract class of raytracer
class cuBVH {
public:
    cuBVH() {}
    virtual ~cuBVH() {}

    virtual void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) = 0;
    virtual void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) = 0;
    virtual void unsigned_distance_backward(at::Tensor grad_dist, at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::Tensor grad_positions) = 0;
    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;
};

// function to create an implementation of cuBVH
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles);

// floodfill
at::Tensor floodfill(at::Tensor grid);
    
} // namespace cubvh