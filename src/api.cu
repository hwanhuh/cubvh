#include <cubvh/api.h>
#include <cubvh/common.h>
#include <cubvh/bvh.cuh>
#include <cubvh/floodfill.cuh>

#include <Eigen/Dense>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

class cuBVHImpl : public cuBVH {
public:

    // accept numpy array (cpu) to init 
    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : cuBVH() {

        const size_t n_vertices = vertices.rows();
        const size_t n_triangles = triangles.rows();

        std::vector<Triangle> triangles_cpu;
        triangles_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2)), (int64_t)i};
        }

        // copy unsorted triangles to GPU
        triangles_unsorted_gpu.resize_and_copy_from_host(triangles_cpu);

        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }

        // build sorts triangles_cpu in-place
        triangle_bvh->build(triangles_cpu, 8);

        // copy sorted triangles to GPU
        triangles_sorted_gpu.resize_and_copy_from_host(triangles_cpu);

    }

    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), triangles_sorted_gpu.data(), stream);
    }

    void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->unsigned_distance_gpu(n_elements, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_sorted_gpu.data(), stream);

    }

    void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->signed_distance_gpu(n_elements, mode, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_sorted_gpu.data(), stream);
    }

    void unsigned_distance_backward(at::Tensor grad_dist, at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::Tensor grad_positions) {
        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->unsigned_distance_backward_gpu(n_elements, grad_dist.data_ptr<float>(), positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), grad_positions.data_ptr<float>(), triangles_unsorted_gpu.data(), stream);
    }

    GPUMemory<Triangle> triangles_sorted_gpu;
    GPUMemory<Triangle> triangles_unsorted_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles};
}

at::Tensor floodfill(at::Tensor grid) {

    // assert grid is uint8_t
    assert(grid.dtype() == at::ScalarType::Bool);

    const int H = grid.size(0);
    const int W = grid.size(1);
    const int D = grid.size(2);

    // allocate mask
    at::Tensor mask = at::zeros({H, W, D}, at::device(grid.device()).dtype(at::ScalarType::Int));

    _floodfill(grid.data_ptr<bool>(), H, W, D, mask.data_ptr<int32_t>());

    return mask;
}

} // namespace cubvh