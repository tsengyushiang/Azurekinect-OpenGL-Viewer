#include "cudaUtils.cuh"
#include <stdio.h>

#define NEIGHBORINDEXSTEP 1

__global__ void planeVertexNormalEstimate_kernel(
    float* vbo_data, unsigned int w, unsigned int h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * w + x;

    if (x < w - 1 && y < h - 1 && x > 0 && y > 0) {
        float center[3] = {
            vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
            vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
            vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
        };

        unsigned int leftindex = y * w + (x - 1);
        float left[3] = {
            vbo_data[leftindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
            vbo_data[leftindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
            vbo_data[leftindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
        };

        unsigned int rightindex = y * w + (x + 1);
        float right[3] = {
            vbo_data[rightindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
            vbo_data[rightindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
            vbo_data[rightindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
        };

        unsigned int topindex = (y - 1) * w + x;
        float top[3] = {
            vbo_data[topindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
            vbo_data[topindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
            vbo_data[topindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
        };

        unsigned int downindex = (y + 1) * w + x;
        float down[3] = {
            vbo_data[downindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
            vbo_data[downindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
            vbo_data[downindex * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
        };

        //Diff to right
        float dx1 = right[0] - left[0];
        float dy1 = right[1] - left[1];
        float dz1 = right[2] - left[2];

        //Diff to bottom
        float dx2 = down[0] - top[0];
        float dy2 = down[1] - top[1];
        float dz2 = down[2] - top[2];

        //d1 cross d2
        float normx = dy1 * dz2 - dz1 * dy2;
        float normy = dz1 * dx2 - dx1 * dz2;
        float normz = dx1 * dy2 - dy1 * dx2;

        //if n dot p > 0, flip towards viewpoint
        if (normx * center[0] + normy * center[1] + normz * center[2] > 0.0f)
        {
            //Flip towards camera
            normx = -normx;
            normy = -normy;
            normz = -normz;
        }

        float length = sqrt(normx * normx + normy * normy + normz * normz);
        vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0] = normx / length;
        vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1] = normy / length;
        vbo_data[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2] = normz / length;
    }    
   
}

void launch_kernel(float* pos, unsigned int w, unsigned int h)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    planeVertexNormalEstimate_kernel << < grid, block >> > (pos, w, h);

}

void CudaAlogrithm::planeVertexNormalEstimate(struct cudaGraphicsResource** vbo_resource,
    unsigned int w, unsigned int h
)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    launch_kernel(dptr, w, h);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}