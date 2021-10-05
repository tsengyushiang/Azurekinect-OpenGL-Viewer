#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void clipFloorAndFarDepth_kernel(
    cudaSurfaceObject_t mask, uint16_t* depthRaw, unsigned int w, unsigned int h, float* xy_table, float depthScale, glm::mat4 modelMat,
    glm::mat4 world2BoundingBoxMat, glm::vec3 boundingboxMax, glm::vec3 boundingboxmin
)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = (h - 1 - y) * w + x;

    uchar4 pixelCenter = { 0,0,0,0 };
    surf2Dread(&pixelCenter, mask, x * sizeof(uchar4), y);

    float depthValue = (float)depthRaw[index] * depthScale;

    glm::vec4 worldPosition = world2BoundingBoxMat * modelMat * glm::vec4(
        xy_table[index * 2] * depthValue,
        xy_table[index * 2 + 1] * depthValue,
        depthValue,
        1.0
    );

    if (worldPosition.x < boundingboxmin.x ||
        worldPosition.y < boundingboxmin.y || 
        worldPosition.z < boundingboxmin.z || 
        worldPosition.x > boundingboxMax.x || 
        worldPosition.y > boundingboxMax.y ||
        worldPosition.z > boundingboxMax.z ) {
        pixelCenter.w = 0;
    }

    surf2Dwrite(pixelCenter, mask, x * sizeof(uchar4), y);
}

void launch_kernel(
    cudaSurfaceObject_t mask, uint16_t* depthRaw,
    unsigned int w, unsigned int h, float* xy_table, float depthScale, glm::mat4 modelMat,
    glm::mat4 world2BoundingBoxMat, glm::vec3 boundingboxMax, glm::vec3 boundingboxmin
)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    clipFloorAndFarDepth_kernel << < grid, block >> > (mask, depthRaw, w, h, xy_table, depthScale, modelMat, world2BoundingBoxMat, boundingboxMax, boundingboxmin);
}

void CudaAlogrithm::boundingboxWorldClipping(
    cudaGraphicsResource_t* maskTexture,
    uint16_t* depthRaw, unsigned int w, unsigned int h, float* xy_table, float depthScale, glm::mat4 modelMat,
    glm::mat4 world2BoundingBoxMat, glm::vec3 boundingboxMax, glm::vec3 boundingboxmin
)
{
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
    launch_kernel(surfObject2, depthRaw, w, h, xy_table, depthScale, modelMat, world2BoundingBoxMat, boundingboxMax, boundingboxmin);

    // We're not going to use this Surface object again.  We'll make a new one next frame.
    cudaDestroySurfaceObject(surfObject2);
    cudaGraphicsUnmapResources(1, maskTexture, 0);
}