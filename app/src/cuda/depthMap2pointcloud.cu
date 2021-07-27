#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthMap2point_kernel(
    float* pos, 
    unsigned short* depthRaw, cudaSurfaceObject_t colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = (h-1-y) * w + x;

    float depthValue = (float)depthRaw[index] * depthScale;

    uchar4 pixel = { 0,0,0,0 };
    surf2Dread(&pixel, colorRaw, x * sizeof(uchar4), y);

    if (depthValue < depthThreshold && pixel.w!=0) {

        // write output vertex
        pos[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0] = (x - ppx) / fx * depthValue;
        pos[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1] = (y - ppy) / fy * depthValue;
        pos[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2] = depthValue;

        pos[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_UV + 0] = float(x) / float(w);
        pos[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_UV + 1] = float(y) / float(h);
    }
    else {
        pos[index * ATTRIBUTESIZE + 0] = 0;
        pos[index * ATTRIBUTESIZE + 1] = 0;
        pos[index * ATTRIBUTESIZE + 2] = 0;
    }

}

void launch_kernel(float* pos,
    unsigned short* depthRaw, cudaSurfaceObject_t colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    depthMap2point_kernel << < grid, block >> > (pos, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);
}

void CudaAlogrithm::depthMap2point(struct cudaGraphicsResource** vbo_resource,
    unsigned short* depthRaw, cudaGraphicsResource_t* cudaTexture,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    cudaArray* texture_ptr;

    cudaGraphicsMapResources(1, cudaTexture, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTexture, 0, 0);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texture_ptr;
    cudaSurfaceObject_t surfObject;
    cudaCreateSurfaceObject(&surfObject, &resDesc);
    // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(dptr, depthRaw, surfObject, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);
    
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}