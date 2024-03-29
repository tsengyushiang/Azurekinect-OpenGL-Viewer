﻿#include "cudaUtils.cuh"
#include <stdio.h>

__device__ bool checkFaceAnglePassThreshold(
    float cosValue,
    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
    glm::vec3 n0, glm::vec3 n1, glm::vec3 n2
) {
    glm::vec3 center = (p0 + p1 + p2) / 3.0f;
    glm::vec3 faceNormal = (n0 + n1 + n2)/3.0f;

    float weight = glm::dot(glm::normalize(-center), faceNormal);

    if (ISVALIDDEPTHVALUE(p0.z) && ISVALIDDEPTHVALUE(p1.z) && ISVALIDDEPTHVALUE(p2.z) &&
        weight > cosValue) {
        return true;
    }
    return false;
}

__device__ bool checkMinAnglePassThreshold(
    float degreeThreshold,
    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2
) {
    float threshold = degreeThreshold / 180 * 3.1415926;
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
    if (ISVALIDDEPTHVALUE(p0.z) && ISVALIDDEPTHVALUE(p1.z) && ISVALIDDEPTHVALUE(p2.z) &&
        angle0 > threshold && angle1 > threshold && angle2 > threshold) {
        return true;
    }
    return false;
}

// triangultae two triangle at one time
__global__ void depthMapTriangulate_kernel(
    float* pos, unsigned int* indices,
    unsigned int width, unsigned int height, int* counter, float degree)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((y + 1) >= height || (x + 1) >= width) return;

    unsigned int index0 = (y * width + x);
    unsigned int index1 = (y * width + (x+1));
    unsigned int index2 = ((y+1) * width + x);
    unsigned int index3 = ((y+1) * width + (x+1));

    glm::vec3 p0 = glm::vec3(
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
    ); 
    glm::vec3 n0 = glm::vec3(
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0],
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1],
        pos[index0 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2]
    );
    glm::vec3 p1 = glm::vec3(
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
    ); 
    glm::vec3 n1 = glm::vec3(
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0],
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1],
        pos[index1 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2]
    );
    glm::vec3 p2 = glm::vec3(
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
    );    
    glm::vec3 n2 = glm::vec3(
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0],
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1],
        pos[index2 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2]
    );
    glm::vec3 p3 = glm::vec3(
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0],
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1],
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
    );
    glm::vec3 n3 = glm::vec3(
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0],
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1],
        pos[index3 * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2]
    );

    // triangulate upper triangle
    //if (checkMinAnglePassThreshold(degree, p0, p1, p2)) {
    //if (checkMinAnglePassThreshold(degree, p0, p1, p2)) {
    if(checkFaceAnglePassThreshold(degree,p0,p1,p2,n0,n1,n2)){
        int i = atomicAdd(counter, 1);
        indices[(i) * 3 + 0] = index0;
        indices[(i) * 3 + 1] = index1;
        indices[(i) * 3 + 2] = index2;
    }

    // triangulate lower triangle
    //if (checkMinAnglePassThreshold(degree, p1, p2, p3)) {
    if (checkFaceAnglePassThreshold(degree, p1, p2,p3,n1, n2,n3)){
        int i2 = atomicAdd(counter, 1);
        indices[(i2) * 3 + 0] = index1;
        indices[(i2) * 3 + 1] = index3;
        indices[(i2) * 3 + 2] = index2;
    }
}

void launch_kernel(float* pos, unsigned int* indices,
    unsigned int mesh_width, unsigned int mesh_height, int* count, float degree)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    depthMapTriangulate_kernel << < grid, block >> > (pos, indices, mesh_width, mesh_height, count, degree);
}

void CudaAlogrithm::depthMapTriangulate(
    struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
    unsigned int w, unsigned int h, int* count, float degree)
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
    launch_kernel(dptr, dptr2, w, h, count, degree);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, ibo_resource, 0);
}