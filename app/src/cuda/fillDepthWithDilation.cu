#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthErosion_kernel(
    bool* hasChangedMask, unsigned short* depthIn, unsigned short* depthOut,unsigned int w, unsigned int h, int dilationPixel)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    unsigned short depth = depthIn[index];
    depthOut[index] = depth;
    // in mask but have no depth
    if (hasChangedMask[index]) {
        // go through neighbor
        for (int shiftX = -dilationPixel; shiftX <= dilationPixel; shiftX++) {
            for (int shiftY = -dilationPixel; shiftY <= dilationPixel; shiftY++) {

                if (
                    shiftX != 0 &&
                    shiftY != 0 &&
                    (y + shiftY) > 0 &&
                    (y + shiftY) < h &&
                    (shiftX + x) > 0 &&
                    (shiftX + x) < w
                    )
                {
                    int indexNeighbor = (y + shiftY) * w + (shiftX + x);
                    unsigned short depthNeighbor = depthIn[indexNeighbor];

                    // if neighbor is in mask and have depth
                    if (!ISVALIDDEPTHVALUE(depthNeighbor)) {
                        // use the depth value
                        depthOut[index] = 0;
                        return;
                    }
                }
            }
        }
        return;
    }
}

__global__ void depthDilation_kernel(
    bool* hasChangedMask,unsigned short* depthIn, unsigned short* depthOut,
    unsigned int w, unsigned int h, int dilationPixel)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    unsigned short depth = depthIn[index];
    hasChangedMask[index] = false;
    depthOut[index] = depth;
    // in mask but have no depth
    if (!ISVALIDDEPTHVALUE(depth)) {
        // go through neighbor
        for (int shiftX = -dilationPixel; shiftX <= dilationPixel; shiftX++) {
            for (int shiftY = -dilationPixel; shiftY <= dilationPixel; shiftY++) {

                if (
                    shiftX !=0 &&
                    shiftY !=0 &&
                    (y + shiftY) > 0 &&
                    (y + shiftY) < h &&
                    (shiftX + x) > 0 &&
                    (shiftX + x) < w
                    )
                {
                    int indexNeighbor = (y + shiftY) * w + (shiftX + x);
                    unsigned short depthNeighbor = depthIn[indexNeighbor];

                    // if neighbor is in mask and have depth
                    if (ISVALIDDEPTHVALUE(depthNeighbor)) {
                        // use the depth value
                        depthOut[index] = depthNeighbor;
                        hasChangedMask[index] = true;
                        return;
                    }
                }
            }
        }
        return;
    }
}

void launch_kernel(unsigned short* depth, unsigned short* dilatedDepth, unsigned int w, unsigned int h, int interation)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);

    bool* changedMask;
    cudaMalloc((void**)&changedMask, w * h * sizeof(bool));

    depthDilation_kernel << < grid, block >> > (changedMask,depth, dilatedDepth, w, h, interation);
    depthErosion_kernel << < grid, block >> > (changedMask,dilatedDepth, depth, w, h, interation);

    cudaFree(changedMask);
}

void CudaAlogrithm::fillDepthWithDilation(cudaGraphicsResource_t* mask,
    unsigned short* depthRaw, unsigned short* dilatedDepth, unsigned int w, unsigned int h, int dilationPixel)
{
    launch_kernel(depthRaw, dilatedDepth, w, h, dilationPixel);
}