#pragma once
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../config.h"

class CudaAlogrithm {
public:
	static void depthMap2point(struct cudaGraphicsResource** vbo_resource,
		unsigned short* depthRaw, cudaGraphicsResource_t* cudaTexture,
		unsigned int w, unsigned int h, float* xy_table, float depthScale, float depthThreshold
	);
		
	static void planePointsLaplacianSmoothing(struct cudaGraphicsResource** vbo_resource,
		unsigned int w, unsigned int h, int interation
	);

	static void planeVertexNormalEstimate(struct cudaGraphicsResource** vbo_resource,
		unsigned int w, unsigned int h
	);

	static void depthMapTriangulate(
		struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
		unsigned int w, unsigned int h,int *count,float degree
	);

	static void chromaKeyBackgroundRemove(cudaGraphicsResource_t* cudaTexture,
		unsigned char* colorRaw, unsigned int w, unsigned int h, glm::vec3 color, glm::vec3 HSVthreshold
	);	
	
	static void chromaKeyBackgroundRemove(cudaGraphicsResource_t* cudaTexture,
		unsigned char* colorRaw, unsigned int w, unsigned int h, glm::vec3 color, glm::vec3 HSVthreshold, glm::vec3 replaceColor
	);

	static void maskErosion(cudaGraphicsResource_t* cudaTexture, 
		unsigned int w, unsigned int h, int erosionPixel
	);

	static void fillDepthWithDilation(cudaGraphicsResource_t* mask, 
		unsigned short* depthRaw, unsigned int w, unsigned int h
	);

	static void depthVisualize(cudaGraphicsResource_t* mask, cudaGraphicsResource_t* cudaTexture,
		uint16_t* colorRaw, unsigned int w, unsigned int h, float depthScale,float farplane);	

	static void clipFloorAndFarDepth(cudaGraphicsResource_t* mask,
		uint16_t* depthRaw, unsigned int w, unsigned int h, float* xy_table, float depthScale, float farplane,
		glm::vec3 planeCenter, glm::vec3 planeNormal,float planeCullingDistance);

	static void boundingboxWorldClipping(cudaGraphicsResource_t* mask,
		uint16_t* depthRaw, unsigned int w, unsigned int h, float* xy_table, float depthScale, glm::mat4 modelMat,
		glm::mat4 boundingboxWorld,glm::vec3 boundingboxMax, glm::vec3 boundingboxmin
	);

};