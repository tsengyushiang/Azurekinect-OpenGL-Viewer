#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthVisualize_kernel(
    cudaSurfaceObject_t mask, uint16_t* depthRaw, cudaSurfaceObject_t output, unsigned int w, unsigned int h,float depthScale, float far)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    uchar4 pixelCenter = { 0,0,0,0 };
    surf2Dread(&pixelCenter, mask, x * sizeof(uchar4), y);
    float depth = float(depthRaw[index]) * depthScale / far * 255;
    bool isCulled = (depth > 255) || (pixelCenter.w == 0);

    uchar4 pixel = {
        isCulled ? 255:depth,
        isCulled ? 0 : depth,
        isCulled ? 0 : depth,
        1.0
    };
    //Write the new pixel color to the 
    surf2Dwrite(pixel, output, x * sizeof(uchar4), y);
}

void launch_kernel(
    cudaSurfaceObject_t mask, uint16_t* depthRaw, cudaSurfaceObject_t output, unsigned int w, unsigned int h, float depthScale, float far)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    depthVisualize_kernel << < grid, block >> > (mask, depthRaw, output, w, h, depthScale, far);
}

void CudaAlogrithm::depthVisualize(
    cudaGraphicsResource_t*maskTexture, cudaGraphicsResource_t* cudaTexture,
    uint16_t* depthRaw, unsigned int w, unsigned int h, float depthScale, float far)
{
    cudaArray* texture_ptr;

    cudaGraphicsMapResources(1, cudaTexture, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTexture, 0, 0);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texture_ptr;
    cudaSurfaceObject_t surfObject;
    cudaCreateSurfaceObject(&surfObject, &resDesc);

    cudaArray* mask_ptr;

    cudaGraphicsMapResources(1, maskTexture, 0);
    cudaGraphicsSubResourceGetMappedArray(&mask_ptr, *maskTexture, 0, 0);

    cudaResourceDesc resDesc2;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = cudaResourceTypeArray;
    resDesc2.res.array.array = mask_ptr;
    cudaSurfaceObject_t surfObject2;
    cudaCreateSurfaceObject(&surfObject2, &resDesc2);
    // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(surfObject2, depthRaw, surfObject, w, h, depthScale, far);

    // We're not going to use this Surface object again.  We'll make a new one next frame.
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);
}