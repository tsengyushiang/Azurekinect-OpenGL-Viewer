#include "cudaUtils.cuh"
#include <stdio.h>

#define MAXITERATION 50
#define HOLETHRESHOLD 1e-3

__global__ void depthDilation_kernel(
    cudaSurfaceObject_t mask, unsigned short* depthIn, unsigned short* depthOut,
    unsigned int w, unsigned int h, int* done)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    uchar4 pixelCenter = { 0,0,0,0 };
    surf2Dread(&pixelCenter, mask, x * sizeof(uchar4), y);
    unsigned short depth = depthIn[index];

    depthOut[index] = depth;

    // in mask but have no depth
    if (pixelCenter.w > 0 && depth < HOLETHRESHOLD) {

        // go through neighbor
        int erosionPixel = 1;
        for (int shiftX = -erosionPixel; shiftX <= erosionPixel; shiftX++) {
            for (int shiftY = -erosionPixel; shiftY <= erosionPixel; shiftY++) {

                if (
                    shiftX !=0 &&
                    shiftY !=0 &&
                    (y + shiftY) > 0 &&
                    (y + shiftY) < h &&
                    (shiftX + x) > 0 &&
                    (shiftX + x) < w
                    )
                {
                    uchar4 pixelNeighbor = { 0,0,0,0 };
                    surf2Dread(&pixelNeighbor, mask, (shiftX + x) * sizeof(uchar4), (y + shiftY));
                    int indexNeighbor = (y + shiftY) * w + (shiftX + x);
                    unsigned short depthNeighbor = depthIn[indexNeighbor];

                    // if neighbor is in mask and have depth
                    if (pixelNeighbor.w > 0 && depthNeighbor > HOLETHRESHOLD) {                        
                        // use the depth value
                        depthOut[index] = depthNeighbor;
                        return;
                    }
                }
            }
        }

        // if all neighbor in mask has no depth, fill out in next iteration
        done = 0;
        return;
    }
}

void launch_kernel(cudaSurfaceObject_t mask, unsigned short* depth, unsigned int w, unsigned int h)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);

    bool dstIstmp = true;
    uint16_t* tmpDepthMap;
    cudaMalloc((void**)&tmpDepthMap, w * h * sizeof(uint16_t));

    int allPixelisReady = 1;
    cudaMalloc((void**)&allPixelisReady, sizeof(int));


    for (int i = 0; i < MAXITERATION; i++) {
        if (dstIstmp) {
            depthDilation_kernel << < grid, block >> > (mask, depth, tmpDepthMap, w, h, &allPixelisReady);
        }
        else {
            depthDilation_kernel << < grid, block >> > (mask, tmpDepthMap, depth, w, h, &allPixelisReady);
        }
        dstIstmp = !dstIstmp;
        if (allPixelisReady == 1) {
            printf("finish dilation in iteration : %d",i);
            break;
        }
    }
    if (!dstIstmp) {
        cudaMemcpy(tmpDepthMap, depth, w * h * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    }

    cudaFree(tmpDepthMap);
    cudaFree(&allPixelisReady);
}

void CudaAlogrithm::fillDepthWithDilation(cudaGraphicsResource_t* mask,
    unsigned short* depthRaw, unsigned int w, unsigned int h)
{
    cudaArray* texture_ptr;

    cudaGraphicsMapResources(1, mask, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *mask, 0, 0);

    cudaResourceDesc origion;
    memset(&origion, 0, sizeof(origion));
    origion.resType = cudaResourceTypeArray;
    origion.res.array.array = texture_ptr;
    cudaSurfaceObject_t surfObject;
    cudaCreateSurfaceObject(&surfObject, &origion);

     // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(surfObject, depthRaw, w, h);

    // We're not going to use this Surface object again.  We'll make a new one next frame.
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, mask, 0);
}