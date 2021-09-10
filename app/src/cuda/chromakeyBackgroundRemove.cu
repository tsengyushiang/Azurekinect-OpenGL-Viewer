#include "cudaUtils.cuh"
#include <stdio.h>
#include "../InputCamera/InputBase.h"

__device__  void RGBtoHSV(float*fR, float*fG, float *fB, float*fH, float*fS, float*fV) {
    float fCMax = max(max(*fR, *fG), *fB);
    float fCMin = min(min(*fR, *fG), *fB);
    float fDelta = fCMax - fCMin;

    if (fDelta > 0) {
        if (fCMax == *fR) {
            *fH = 60 * (fmod(((*fG - *fB) / fDelta), float(6)));
        }
        else if (fCMax == *fG) {
            *fH = 60 * (((*fB - *fR) / fDelta) + 2);
        }
        else if (fCMax == *fB) {
            *fH = 60 * (((*fR - *fG) / fDelta) + 4);
        }

        if (fCMax > 0) {
            *fS = fDelta / fCMax;
        }
        else {
            *fS = 0;
        }

        *fV = fCMax;
    }
    else {
        *fH = 0;
        *fS = 0;
        *fV = fCMax;
    }

    if (*fH < 0) {
        *fH = 360 + *fH;
    }
}

__global__ void chromaKeyBackgroundRemove_kernel(
    unsigned char* colorRaw, cudaSurfaceObject_t output, unsigned int w, unsigned int h, glm::vec3 chromakey, glm::vec3 HSVthreshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = (h-1-y) * w + x;

    glm::vec3 color = glm::vec3(
        colorRaw[index * INPUT_COLOR_CHANNEL + 2],
        colorRaw[index * INPUT_COLOR_CHANNEL + 1],
        colorRaw[index * INPUT_COLOR_CHANNEL + 0]
    );

    float cH, cS, cV;
    RGBtoHSV(&chromakey.x, &chromakey.y, &chromakey.z, &cH, &cS, &cV);
    float H, S, V;
    RGBtoHSV(&color.x, &color.y, &color.z,&H,&S,&V);
    int alpha = 255;

    if ((abs(cH / 360.0 * 255.0 -H / 360.0 * 255.0) < HSVthreshold.x) ||
        (abs(cS*255 - S*255) < HSVthreshold.y) ||
        (abs(cV - V) < HSVthreshold.z)) {
        alpha = 0;
    }
    uchar4 pixel = {
       colorRaw[index * INPUT_COLOR_CHANNEL + 2],
        colorRaw[index * INPUT_COLOR_CHANNEL + 1],
        colorRaw[index * INPUT_COLOR_CHANNEL + 0],
        alpha
    };
    //Write the new pixel color to the 
    surf2Dwrite(pixel, output, x * sizeof(uchar4), y);
}

__global__ void chromaKeyBackgroundRemove_kernel(
    unsigned char* colorRaw, cudaSurfaceObject_t output, unsigned int w, unsigned int h, glm::vec3 chromakey, glm::vec3 HSVthreshold, glm::vec3 replaceColor)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = (h-1-y) * w + x;

    glm::vec3 color = glm::vec3(
        colorRaw[index * INPUT_COLOR_CHANNEL + 2],
        colorRaw[index * INPUT_COLOR_CHANNEL + 1],
        colorRaw[index * INPUT_COLOR_CHANNEL + 0]
    );

    float cH, cS, cV;
    RGBtoHSV(&chromakey.x, &chromakey.y, &chromakey.z, &cH, &cS, &cV);
    float H, S, V;
    RGBtoHSV(&color.x, &color.y, &color.z, &H, &S, &V);
    int alpha = 255;

    if (abs(cH / 360.0 * 255.0 - H / 360.0 * 255.0) < HSVthreshold.x ||
        abs(cS * 255 - S * 255) < HSVthreshold.y ||
        abs(cV - V) < HSVthreshold.z) {
        alpha = 0;
    }

    uchar4 pixel = {
        replaceColor.x,
        replaceColor.y,
        replaceColor.z,
        alpha
    };
    //Write the new pixel color to the 
    surf2Dwrite(pixel, output, x * sizeof(uchar4), y);
}

void launch_kernel(
    unsigned char* colorRaw, cudaSurfaceObject_t output, unsigned int mesh_width, unsigned int mesh_height, glm::vec3 color, glm::vec3 HSVthreshold)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    chromaKeyBackgroundRemove_kernel << < grid, block >> > (colorRaw, output, mesh_width, mesh_height, color, HSVthreshold);
}

void launch_kernel(
    unsigned char* colorRaw, cudaSurfaceObject_t output, unsigned int mesh_width, unsigned int mesh_height, glm::vec3 color, glm::vec3 HSVthreshold, glm::vec3 replaceColor)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    chromaKeyBackgroundRemove_kernel << < grid, block >> > (colorRaw, output, mesh_width, mesh_height, color, HSVthreshold, replaceColor);
}

void CudaAlogrithm::chromaKeyBackgroundRemove(
    cudaGraphicsResource_t* cudaTexture,
    unsigned char* colorRaw, unsigned int w, unsigned int h,glm::vec3 color, glm::vec3 HSVthreshold)
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
    // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(colorRaw, surfObject, w, h, color, HSVthreshold);

    // We're not going to use this Surface object again.  We'll make a new one next *fRame.
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);
}

void CudaAlogrithm::chromaKeyBackgroundRemove(
    cudaGraphicsResource_t* cudaTexture,
    unsigned char* colorRaw, unsigned int w, unsigned int h, glm::vec3 color, glm::vec3 HSVthreshold,glm::vec3 replaceColor)
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
    // You now have a CUDA Surface object that refers to the GL texture.
    // Write to the Surface using CUDA.
    launch_kernel(colorRaw, surfObject, w, h, color, HSVthreshold, replaceColor);

    // We're not going to use this Surface object again.  We'll make a new one next *fRame.
    cudaDestroySurfaceObject(surfObject);
    cudaGraphicsUnmapResources(1, cudaTexture, 0);
}