#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void maskErosion_kernel(
    cudaSurfaceObject_t input, cudaSurfaceObject_t output,
    unsigned int w, unsigned int h, int erosionPixel)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 pixelCenter = { 0,0,0,0 };
    surf2Dread(&pixelCenter, input, x * sizeof(uchar4), y);

    if (pixelCenter.w > 0) {
        for (int shiftX = -erosionPixel; shiftX <= erosionPixel; shiftX++) {
            for (int shiftY = -erosionPixel; shiftY <= erosionPixel; shiftY++) {

                if (
                    (y + shiftY) > 0 &&
                    (y + shiftY) < h &&
                    (shiftX + x) > 0 &&
                    (shiftX + x) < w
                    ) 
                {

                    uchar4 pixelNeighbor = { 0,0,0,0 };
                    surf2Dread(&pixelNeighbor, input, (shiftX + x) * sizeof(uchar4), (y + shiftY));

                    if (pixelNeighbor.w == 0) {
                        pixelCenter.w = 0;
                        surf2Dwrite(pixelCenter, output, x * sizeof(uchar4), y);
                        return;
                    }
                }
            }
        }
    }
}

void launch_kernel(cudaSurfaceObject_t input,cudaSurfaceObject_t output, unsigned int w, unsigned int h, int erosionPixel)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    maskErosion_kernel << < grid, block >> > (input,output, w, h, erosionPixel);
}

void CudaAlogrithm::maskErosion(cudaGraphicsResource_t* cudaTexture, unsigned int w, unsigned int h, int erosionPixel)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, w, h, cudaArraySurfaceLoadStore);

    cudaArray* texture_ptr;

    cudaGraphicsMapResources(1, cudaTexture, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTexture, 0, 0);

    cudaMemcpyArrayToArray(cuInputArray, 0, 0, texture_ptr, 0,0, w * h * sizeof(uchar4));

    cudaResourceDesc origion;
    memset(&origion, 0, sizeof(origion));
    origion.resType = cudaResourceTypeArray;
    origion.res.array.array = texture_ptr;
    cudaSurfaceObject_t surfObject;
    cudaCreateSurfaceObject(&surfObject, &origion);

    cudaResourceDesc copyinput;
    memset(&copyinput, 0, sizeof(copyinput));
    copyinput.resType = cudaResourceTypeArray;
    copyinput.res.array.array = cuInputArray;
    cudaSurfaceObject_t surfObjectCopy;
    cudaCreateSurfaceObject(&surfObjectCopy, &copyinput);

    // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(surfObjectCopy,surfObject, w, h, erosionPixel);

    // We're not going to use this Surface object again.  We'll make a new one next frame.
    cudaDestroySurfaceObject(surfObject);
    cudaDestroySurfaceObject(surfObjectCopy);
    cudaFreeArray(cuInputArray);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);
}