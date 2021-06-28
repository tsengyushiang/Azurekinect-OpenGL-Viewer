#pragma once
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class CudaAlogrithm {
public:
	static void depthMap2point(struct cudaGraphicsResource** vbo_resource,
		unsigned short* depthRaw, cudaGraphicsResource_t* cudaTexture,
		unsigned int w, unsigned int h,
		float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold,
		int DilationErosionIteration
	);

	static void depthMap2point(struct cudaGraphicsResource** vbo_resource,
		unsigned short* depthRaw, unsigned char* colorRaw,
		unsigned int w, unsigned int h,
		float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold,
		int DilationErosionIteration
	);
	static void depthMapTriangulate(
		struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
		unsigned int w, unsigned int h,int *count,float degree
	);

	static void chromaKeyBackgroundRemove(cudaGraphicsResource_t* cudaTexture,
		unsigned char* colorRaw, unsigned int w, unsigned int h, glm::vec3 color, float threshold
	);	
	
	static void chromaKeyBackgroundRemove(cudaGraphicsResource_t* cudaTexture,
		unsigned char* colorRaw, unsigned int w, unsigned int h, glm::vec3 color, float threshold, glm::vec3 replaceColor
	);

	static void maskErosion(cudaGraphicsResource_t* cudaTexture, 
		unsigned int w, unsigned int h, int erosionPixel
	);

	static void fillDepthWithDilation(cudaGraphicsResource_t* mask, 
		unsigned short* depthRaw, unsigned int w, unsigned int h
	);
};