#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthMap2point_kernel(
    float* pos, 
    unsigned short* depthRaw, unsigned char* colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    float depthValue = (float)depthRaw[index] * depthScale;
    if (depthValue < depthThreshold) {
        // write output vertex
        pos[index * 6 + 0] = (x - ppx) / fx * depthValue;
        pos[index * 6 + 1] = (y - ppy) / fy * depthValue;
        pos[index * 6 + 2] = depthValue;

        pos[index * 6 + 3] = (float)colorRaw[index * 3 + 2] / 255;
        pos[index * 6 + 4] = (float)colorRaw[index * 3 + 1] / 255;
        pos[index * 6 + 5] = (float)colorRaw[index * 3 + 0] / 255;
    }
    else {
        pos[index * 6 + 0] = 0;
        pos[index * 6 + 1] = 0;
        pos[index * 6 + 2] = 0;
    }

}

void launch_kernel(float* pos, 
    unsigned short* depthRaw, unsigned char* colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    depthMap2point_kernel << < grid, block >> > (pos, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);
}

void CudaAlogrithm::depthMap2point(struct cudaGraphicsResource** vbo_resource,
    unsigned short* depthRaw, unsigned char* colorRaw,
    unsigned int w,unsigned int h,
    float fx,float fy,float ppx,float ppy, float depthScale, float depthThreshold)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    launch_kernel(dptr, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}