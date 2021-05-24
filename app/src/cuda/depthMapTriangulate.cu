#include "cudaUtils.cuh"
#include <stdio.h>

__global__ void depthMapTriangulate_kernel(
    float* pos, unsigned int* indices,
    unsigned int width, unsigned int height, int* counter)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((y + 1) >= height || (x + 1) >= width) return;

    unsigned int index0 = (y * width + x);
    unsigned int index1 = (y * width + (x+1));
    unsigned int index2 = ((y+1) * width + x);
    unsigned int index3 = ((y+1) * width + (x+1));

    glm::vec3 p0 = glm::vec3(
        pos[index0 * 6 + 0],
        pos[index0 * 6 + 1],
        pos[index0 * 6 + 2]
    ); 
    glm::vec3 p1 = glm::vec3(
        pos[index1 * 6 + 0],
        pos[index1 * 6 + 1],
        pos[index1 * 6 + 2]
    ); 
    glm::vec3 p2 = glm::vec3(
        pos[index2 * 6 + 0],
        pos[index2 * 6 + 1],
        pos[index2 * 6 + 2]
    );    
    glm::vec3 p3 = glm::vec3(
        pos[index3 * 6 + 0],
        pos[index3 * 6 + 1],
        pos[index3 * 6 + 2]
    );

    float threshold = 3.1415926 / 16.0;
    glm::vec3 da, db;
    float angle0, angle1, angle2;

    da = glm::normalize(p1 - p0);
    db = glm::normalize(p2 - p0);
    angle0 = glm::acos(glm::dot(da, db));

    da = glm::normalize(p0 - p1);
    db = glm::normalize(p2 - p1);
    angle1 = glm::acos(glm::dot(da, db));

    da = glm::normalize(p0 - p2);
    db = glm::normalize(p1 - p2);
    angle2 = glm::acos(glm::dot(da, db));

    if (angle0 > threshold && angle1 > threshold && angle2 > threshold) {
        int i = atomicAdd(counter, 1);
        indices[(i) * 3 + 0] = index0;
        indices[(i) * 3 + 1] = index2;
        indices[(i) * 3 + 2] = index1;
    }

    da = glm::normalize(p2 - p1);
    db = glm::normalize(p3 - p1);
    angle0 = glm::acos(glm::dot(da, db));

    da = glm::normalize(p1 - p2);
    db = glm::normalize(p3 - p2);
    angle1 = glm::acos(glm::dot(da, db));

    da = glm::normalize(p1 - p3);
    db = glm::normalize(p2 - p3);
    angle2 = glm::acos(glm::dot(da, db));
    if (angle0 > threshold && angle1 > threshold && angle2 > threshold) {
        int i2 = atomicAdd(counter, 1);
        indices[(i2) * 3 + 0] = index1;
        indices[(i2) * 3 + 1] = index2;
        indices[(i2) * 3 + 2] = index3;
    }
}

void launch_kernel(float* pos, unsigned int* indices,
    unsigned int mesh_width, unsigned int mesh_height, int* count)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    depthMapTriangulate_kernel << < grid, block >> > (pos, indices, mesh_width, mesh_height, count);
}

void CudaAlogrithm::depthMapTriangulate(
    struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
    unsigned int w, unsigned int h, int* count)
{
    // map OpenGL buffer object for writing from CUDA
    float* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    // map OpenGL buffer object for writing from CUDA
    unsigned int* dptr2;
    cudaGraphicsMapResources(1, ibo_resource, 0);
    size_t num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr2, &num_bytes2, *ibo_resource);

    cudaMemset(count, 0, sizeof(int));
    launch_kernel(dptr, dptr2, w, h, count);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, ibo_resource, 0);
}