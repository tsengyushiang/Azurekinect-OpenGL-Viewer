#include "cudaUtils.cuh"
#include <stdio.h>

#define NEIGHBORINDEXSTEP 1

__global__ void planePointsLaplacianSmoothing_kernel(
    float* inputPoints,float* outputPoints, unsigned int w, unsigned int h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    for (int i = 0; i < 6; i++) {
        outputPoints[index * 6 + i] = inputPoints[index * 6 + i];
    }
    // self is valid point
    if (inputPoints[index * 6 + 2] != 0) {
        float coordinateSum[3] = { 0,0,0 };
        int validNeighborCount = 0;
        for (int shiftY = -NEIGHBORINDEXSTEP; shiftY <= NEIGHBORINDEXSTEP; shiftY++) {
            for (int shiftX = -NEIGHBORINDEXSTEP; shiftX <= NEIGHBORINDEXSTEP; shiftX++) {

                if ((y + shiftY) > 0 &&
                    (y + shiftY) < h &&
                    (shiftX + x) > 0 &&
                    (shiftX + x) < w
                    )
                {
                    int indexNeighbor = (y + shiftY) * w + (shiftX + x);

                    // neighbor is valid point
                    if (inputPoints[indexNeighbor * 6 + 2] != 0) {
                        coordinateSum[0] += inputPoints[indexNeighbor * 6 + 0];
                        coordinateSum[1] += inputPoints[indexNeighbor * 6 + 1];
                        coordinateSum[2] += inputPoints[indexNeighbor * 6 + 2];
                        validNeighborCount++;
                    }
                }
            }
        }
        if (validNeighborCount != 0) {
            outputPoints[index * 6 + 0] = coordinateSum[0] / validNeighborCount;
            outputPoints[index * 6 + 1] = coordinateSum[1] / validNeighborCount;
            outputPoints[index * 6 + 2] = coordinateSum[2] / validNeighborCount;
        }
    }    
}

void launch_kernel(float* pos, unsigned int w, unsigned int h,int iteration)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    {
        bool dstIstmp = true;
        float* tmpPointsArray;
        cudaMalloc((void**)&tmpPointsArray, w * h * 6 * sizeof(float));

        // dilation
        for (int i = 0; i < iteration; i++) {
            if (dstIstmp) {
                planePointsLaplacianSmoothing_kernel << < grid, block >> > (pos, tmpPointsArray, w, h);
            }
            else {
                planePointsLaplacianSmoothing_kernel << < grid, block >> > (tmpPointsArray, pos, w, h);
            }
            dstIstmp = !dstIstmp;
        }

        if (!dstIstmp) {
            cudaMemcpy(tmpPointsArray, pos, w * h * 6 * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
        }
        cudaFree(tmpPointsArray);
    }
}

void CudaAlogrithm::planePointsLaplacianSmoothing(struct cudaGraphicsResource** vbo_resource,
    unsigned int w, unsigned int h, int interation
)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    launch_kernel(dptr, w, h, interation);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}