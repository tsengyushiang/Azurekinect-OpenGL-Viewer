#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthMapTriangulate_kernel(
    float* pos, unsigned int* indices,
    unsigned int width, unsigned int height, int* counter)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((y + 1) >= height || (x + 1) >= width) return;


    unsigned int index0 = (y * width + x);
    unsigned int index1 = (y * width + (x+1));
    unsigned int index2 = ((y+1) * width + x);
    unsigned int index3 = ((y+1) * width + (x+1));

    int i = atomicAdd(counter, 1);
    indices[(i) * 3 + 0] = index0;
    indices[(i) * 3 + 1] = index2;
    indices[(i) * 3 + 2] = index1;

    int i2 = atomicAdd(counter, 1);
    indices[(i2) * 3 + 0] = index1;
    indices[(i2) * 3 + 1] = index2;
    indices[(i2) * 3 + 2] = index3;
}

void launch_kernel(float* pos, unsigned int* indices,
    unsigned int mesh_width, unsigned int mesh_height, int* count)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    depthMapTriangulate_kernel << < grid, block >> > (pos, indices, mesh_width, mesh_height, count);
}

void CudaAlogrithm::depthMapTriangulate(
    struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
    unsigned int w, unsigned int h, int* count)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    // map OpenGL buffer object for writing from CUDA
    unsigned int* dptr2;
    cudaGraphicsMapResources(1, ibo_resource, 0);
    size_t num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr2, &num_bytes2, *ibo_resource);

    cudaMemset(count, 0, sizeof(int));
    launch_kernel(dptr, dptr2, w, h, count);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, ibo_resource, 0);
}