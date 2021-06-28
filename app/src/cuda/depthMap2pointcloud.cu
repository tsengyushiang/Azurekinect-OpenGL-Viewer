#include "cudaUtils.cuh"
#include <stdio.h>

#define ARROUNDPIXELCOUNT 4
#define HOLETHRESHOLD 1e-3

__global__ void erosion_kernel(unsigned short* depthRaw, unsigned short* result, unsigned int w, unsigned int h) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    unsigned int indexAround[ARROUNDPIXELCOUNT] = {
        (y + 1) * w + x,
        (y - 1) * w + x,
        y * w + (x + 1),
        y * w + (x - 1)
    };

    result[index] = depthRaw[index];

    if (depthRaw[index] > HOLETHRESHOLD) {
        for (int i = 0; i < ARROUNDPIXELCOUNT; i++) {
            unsigned int neighbor = indexAround[i];
            if (neighbor > 0 && neighbor < w * h) {
                if (depthRaw[neighbor] < HOLETHRESHOLD) {
                    result[index] = 0;
                    return;
                }
            }
        }
    }
}

__global__ void dilation_kernel(unsigned short* depthRaw, unsigned short* result, unsigned int w, unsigned int h) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    unsigned int indexAround[ARROUNDPIXELCOUNT] = {
        (y + 1) * w + x,
        (y - 1) * w + x,
        y * w + (x + 1),
        y * w + (x - 1)
    };

    result[index] = depthRaw[index];

    if (depthRaw[index] < HOLETHRESHOLD) {
        for (int i = 0; i < ARROUNDPIXELCOUNT; i++) {
            unsigned int neighbor = indexAround[i];
            if (neighbor > 0 && neighbor < w * h) {
                if (depthRaw[neighbor] > HOLETHRESHOLD) {
                    result[index] = depthRaw[neighbor];
                    return;
                }
            }
        }
    }
}

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

__global__ void depthMap2point_kernel(
    float* pos, 
    unsigned short* depthRaw, cudaSurfaceObject_t colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    float depthValue = (float)depthRaw[index] * depthScale;

    uchar4 pixel = { 0,0,0,0 };
    surf2Dread(&pixel, colorRaw, x * sizeof(uchar4), y);

    if (depthValue < depthThreshold && pixel.w!=0) {

        // write output vertex
        pos[index * 6 + 0] = (x - ppx) / fx * depthValue;
        pos[index * 6 + 1] = (y - ppy) / fy * depthValue;
        pos[index * 6 + 2] = depthValue;

        pos[index * 6 + 3] = (float)pixel.z / 255;
        pos[index * 6 + 4] = (float)pixel.y / 255;
        pos[index * 6 + 5] = (float)pixel.x / 255;
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
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold,int DilationErosionIteration)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    {
        bool dstIstmp = true;
        uint16_t* tmpDepthMap;
        cudaMalloc((void**)&tmpDepthMap, w * h * sizeof(uint16_t));

        // dilation
        for (int i = 0; i < DilationErosionIteration; i++) {
            if (dstIstmp) {
                dilation_kernel << < grid, block >> > (depthRaw, tmpDepthMap, w, h);
            }
            else {
                dilation_kernel << < grid, block >> > (tmpDepthMap, depthRaw, w, h);
            }
            dstIstmp = !dstIstmp;
        }

        //erosion
        for (int i = 0; i < DilationErosionIteration; i++) {
            if (dstIstmp) {
                erosion_kernel << < grid, block >> > (depthRaw, tmpDepthMap, w, h);
            }
            else {
                erosion_kernel << < grid, block >> > (tmpDepthMap, depthRaw, w, h);
            }
            dstIstmp = !dstIstmp;
        }
        if (!dstIstmp) {
            cudaMemcpy(tmpDepthMap, depthRaw, w * h * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
        }
        cudaFree(tmpDepthMap);
    }
    depthMap2point_kernel << < grid, block >> > (pos, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);
}

void launch_kernel(float* pos,
    unsigned short* depthRaw, cudaSurfaceObject_t colorRaw,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold, int DilationErosionIteration)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    {
        bool dstIstmp = true;
        uint16_t* tmpDepthMap;
        cudaMalloc((void**)&tmpDepthMap, w * h * sizeof(uint16_t));

        // dilation
        for (int i = 0; i < DilationErosionIteration; i++) {
            if (dstIstmp) {
                dilation_kernel << < grid, block >> > (depthRaw, tmpDepthMap, w, h);
            }
            else {
                dilation_kernel << < grid, block >> > (tmpDepthMap, depthRaw, w, h);
            }
            dstIstmp = !dstIstmp;
        }

        //erosion
        for (int i = 0; i < DilationErosionIteration; i++) {
            if (dstIstmp) {
                erosion_kernel << < grid, block >> > (depthRaw, tmpDepthMap, w, h);
            }
            else {
                erosion_kernel << < grid, block >> > (tmpDepthMap, depthRaw, w, h);
            }
            dstIstmp = !dstIstmp;
        }
        if (!dstIstmp) {
            cudaMemcpy(tmpDepthMap, depthRaw, w * h * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
        }
        cudaFree(tmpDepthMap);
    }
    depthMap2point_kernel << < grid, block >> > (pos, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold);
}

void CudaAlogrithm::depthMap2point(struct cudaGraphicsResource** vbo_resource,
    unsigned short* depthRaw, unsigned char* colorRaw,
    unsigned int w,unsigned int h,
    float fx,float fy,float ppx,float ppy, float depthScale, float depthThreshold, int DilationErosionIteration)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    launch_kernel(dptr, depthRaw, colorRaw, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold, DilationErosionIteration);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void CudaAlogrithm::depthMap2point(struct cudaGraphicsResource** vbo_resource,
    unsigned short* depthRaw, cudaGraphicsResource_t* cudaTexture,
    unsigned int w, unsigned int h,
    float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold, int DilationErosionIteration)
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
    launch_kernel(dptr, depthRaw, surfObject, w, h, fx, fy, ppx, ppy, depthScale, depthThreshold, DilationErosionIteration);
    
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}