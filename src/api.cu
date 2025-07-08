#include <cubvh/api.h>
#include <cubvh/common.h>
#include <cubvh/bvh.cuh>
#include <cubvh/floodfill.cuh>
#include <cubvh/marching_cubes_tables.cuh>

#include <Eigen/Dense>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

class cuBVHImpl : public cuBVH {
public:
    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : cuBVH() {
        const size_t n_triangles = triangles.rows();
        std::vector<Triangle> triangles_cpu(n_triangles);
        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2)), (int64_t)i};
        }
        triangles_unsorted_gpu.resize_and_copy_from_host(triangles_cpu);
        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }
        triangle_bvh->build(triangles_cpu, 8);
        triangles_sorted_gpu.resize_and_copy_from_host(triangles_cpu);
    }

    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {
        triangle_bvh->ray_trace_gpu(rays_o.size(0), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), triangles_sorted_gpu.data(), at::cuda::getCurrentCUDAStream());
    }

    void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) {
        triangle_bvh->unsigned_distance_gpu(positions.size(0), positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_sorted_gpu.data(), at::cuda::getCurrentCUDAStream());
    }

    void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) {
        triangle_bvh->signed_distance_gpu(positions.size(0), mode, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_sorted_gpu.data(), at::cuda::getCurrentCUDAStream());
    }

    void unsigned_distance_backward(at::Tensor grad_dist, at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::Tensor grad_positions) {
        triangle_bvh->unsigned_distance_backward_gpu(positions.size(0), grad_dist.data_ptr<float>(), positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), grad_positions.data_ptr<float>(), triangles_unsorted_gpu.data(), at::cuda::getCurrentCUDAStream());
    }

    GPUMemory<Triangle> triangles_sorted_gpu;
    GPUMemory<Triangle> triangles_unsorted_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles};
}

at::Tensor floodfill(at::Tensor grid) {
    assert(grid.dtype() == at::ScalarType::Bool);
    const int H = grid.size(0), W = grid.size(1), D = grid.size(2);
    at::Tensor mask = at::zeros({H, W, D}, at::device(grid.device()).dtype(at::ScalarType::Int));
    _floodfill(grid.data_ptr<bool>(), H, W, D, mask.data_ptr<int32_t>());
    return mask;
}

// --- Sparse Marching Cubes Implementation ---

__device__ float3 interpolate_vertex(float3 p1, float3 p2, float v1, float v2) {
    float t = (0.0f - v1) / (v2 - v1);
    return make_float3(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y), p1.z + t * (p2.z - p1.z));
}

__global__ void classify_cubes_kernel(
    const int* __restrict__ active_indices,
    const int* __restrict__ active_map,
    const float* __restrict__ sdf,
    int n_active_vertices,
    int res,
    int* __restrict__ vertex_counts,
    int* __restrict__ case_indices)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n_active_vertices) return;

    int ix = active_indices[index * 3 + 0];
    int iy = active_indices[index * 3 + 1];
    int iz = active_indices[index * 3 + 2];

    if (ix >= res - 1 || iy >= res - 1 || iz >= res - 1) {
        vertex_counts[index] = 0;
        return;
    }

    float corner_sdfs[8];
    int corner_coords[8][3] = {
        {ix, iy, iz}, {ix + 1, iy, iz}, {ix + 1, iy + 1, iz}, {ix, iy + 1, iz},
        {ix, iy, iz + 1}, {ix + 1, iy, iz + 1}, {ix + 1, iy + 1, iz + 1}, {ix, iy + 1, iz + 1}
    };

    bool all_corners_active = true;
    for (int i = 0; i < 8; ++i) {
        int flat_map_idx = corner_coords[i][0] + corner_coords[i][1] * res + corner_coords[i][2] * res * res;
        int sdf_idx = active_map[flat_map_idx];
        if (sdf_idx == -1) {
            all_corners_active = false;
            break;
        }
        corner_sdfs[i] = sdf[sdf_idx];
    }

    if (!all_corners_active) {
        vertex_counts[index] = 0;
        return;
    }

    int case_idx = 0;
    if (corner_sdfs[0] < 0) case_idx |= 1;
    if (corner_sdfs[1] < 0) case_idx |= 2;
    if (corner_sdfs[2] < 0) case_idx |= 4;
    if (corner_sdfs[3] < 0) case_idx |= 8;
    if (corner_sdfs[4] < 0) case_idx |= 16;
    if (corner_sdfs[5] < 0) case_idx |= 32;
    if (corner_sdfs[6] < 0) case_idx |= 64;
    if (corner_sdfs[7] < 0) case_idx |= 128;

    case_indices[index] = case_idx;
    vertex_counts[index] = numVertsTable[case_idx];
}

__global__ void generate_triangles_kernel(
    const float3* __restrict__ vertices,
    const float* __restrict__ sdf,
    const int* __restrict__ active_indices,
    const int* __restrict__ active_map,
    const int* __restrict__ vertex_offsets,
    const int* __restrict__ case_indices,
    int n_active_vertices,
    int res,
    float3* __restrict__ out_vertices,
    int3* __restrict__ out_faces)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n_active_vertices) return;

    int vert_count = (index == 0) ? vertex_offsets[0] : (vertex_offsets[index] - vertex_offsets[index - 1]);
    if (vert_count == 0) return;

    int case_idx = case_indices[index];
    int vert_offset = (index == 0) ? 0 : vertex_offsets[index - 1];

    int ix = active_indices[index * 3 + 0];
    int iy = active_indices[index * 3 + 1];
    int iz = active_indices[index * 3 + 2];

    float3 corner_verts[8];
    float corner_sdfs[8];
    int corner_coords[8][3] = {
        {ix, iy, iz}, {ix + 1, iy, iz}, {ix + 1, iy + 1, iz}, {ix, iy + 1, iz},
        {ix, iy, iz + 1}, {ix + 1, iy, iz + 1}, {ix + 1, iy + 1, iz + 1}, {ix, iy + 1, iz + 1}
    };

    for (int i = 0; i < 8; ++i) {
        int flat_map_idx = corner_coords[i][0] + corner_coords[i][1] * res + corner_coords[i][2] * res * res;
        int sdf_idx = active_map[flat_map_idx];
        corner_verts[i] = vertices[sdf_idx];
        corner_sdfs[i] = sdf[sdf_idx];
    }

    const int* table_row = triTable[case_idx];
    int edge_to_vert[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0}, {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    for (int i = 0; i < vert_count; i += 3) {
        int3 face;
        int edge1 = table_row[i];
        int edge2 = table_row[i+1];
        int edge3 = table_row[i+2];

        int v1_a = edge_to_vert[edge1][0], v1_b = edge_to_vert[edge1][1];
        int v2_a = edge_to_vert[edge2][0], v2_b = edge_to_vert[edge2][1];
        int v3_a = edge_to_vert[edge3][0], v3_b = edge_to_vert[edge3][1];

        out_vertices[vert_offset + i]     = interpolate_vertex(corner_verts[v1_a], corner_verts[v1_b], corner_sdfs[v1_a], corner_sdfs[v1_b]);
        out_vertices[vert_offset + i + 1] = interpolate_vertex(corner_verts[v2_a], corner_verts[v2_b], corner_sdfs[v2_a], corner_sdfs[v2_b]);
        out_vertices[vert_offset + i + 2] = interpolate_vertex(corner_verts[v3_a], corner_verts[v3_b], corner_sdfs[v3_a], corner_sdfs[v3_b]);
        
        face.x = vert_offset + i;
        face.y = vert_offset + i + 1;
        face.z = vert_offset + i + 2;
        out_faces[(vert_offset + i) / 3] = face;
    }
}

void sparse_marching_cubes(
    at::Tensor vertices,
    at::Tensor sdf,
    at::Tensor active_indices,
    at::Tensor active_map,
    int res,
    at::Tensor out_vertices,
    at::Tensor out_faces,
    at::Tensor counters)
{
    const int n_active_vertices = active_indices.size(0);
    if (n_active_vertices == 0) {
        counters[0] = 0;
        counters[1] = 0;
        return;
    }

    const int threads = 256;
    const int blocks = (n_active_vertices + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    at::Tensor vertex_counts = at::empty({n_active_vertices}, vertices.options().dtype(at::kInt));
    at::Tensor case_indices = at::empty({n_active_vertices}, vertices.options().dtype(at::kInt));
    at::Tensor vertex_offsets = at::empty({n_active_vertices}, vertices.options().dtype(at::kInt));

    classify_cubes_kernel<<<blocks, threads, 0, stream>>>(
        active_indices.data_ptr<int>(), active_map.data_ptr<int>(), sdf.data_ptr<float>(),
        n_active_vertices, res, vertex_counts.data_ptr<int>(), case_indices.data_ptr<int>());

    thrust::device_ptr<int> counts_ptr = thrust::device_pointer_cast(vertex_counts.data_ptr<int>());
    thrust::device_ptr<int> offsets_ptr = thrust::device_pointer_cast(vertex_offsets.data_ptr<int>());
    thrust::exclusive_scan(counts_ptr, counts_ptr + n_active_vertices, offsets_ptr);

    int total_verts;
    cudaMemcpy(&total_verts, offsets_ptr.get() + n_active_vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int last_count;
    cudaMemcpy(&last_count, counts_ptr.get() + n_active_vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
    total_verts += last_count;

    int total_faces = total_verts / 3;
    counters[0] = total_verts;
    counters[1] = total_faces;

    out_vertices.resize_({total_verts, 3});
    out_faces.resize_({total_faces, 3});

    if (total_verts > 0) {
        generate_triangles_kernel<<<blocks, threads, 0, stream>>>(
            (const float3*)vertices.data_ptr<float>(), sdf.data_ptr<float>(),
            active_indices.data_ptr<int>(), active_map.data_ptr<int>(),
            vertex_offsets.data_ptr<int>(), case_indices.data_ptr<int>(),
            n_active_vertices, res,
            (float3*)out_vertices.data_ptr<float>(), (int3*)out_faces.data_ptr<int>());
    }
}

} // namespace cubvh
