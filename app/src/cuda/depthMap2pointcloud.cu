#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthMap2point_kernel(float* pos, unsigned int width, unsigned int height, float depthScale)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;

    // write output vertex
    pos[(y * width + x)*6 + 0] = u * 2.0f - 1.0f;
    pos[(y * width + x)*6 + 1] = v * 2.0f - 1.0f;
    pos[(y * width + x)*6 + 2] = depthScale;

    pos[(y * width + x)*6 + 3] = u;
    pos[(y * width + x)*6 + 4] = v;
    pos[(y * width + x)*6 + 5] = 1.0;
}

void launch_kernel(float* pos, unsigned int mesh_width,
    unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    depthMap2point_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
}

void CudaAlogrithm::depthMap2point(struct cudaGraphicsResource** vbo_resource,
    unsigned int w,unsigned int h,float depthScale)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    launch_kernel(dptr, w, h, depthScale);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}