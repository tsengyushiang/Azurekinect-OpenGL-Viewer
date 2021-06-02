#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class CudaAlogrithm {
public:
	static void depthMap2point(struct cudaGraphicsResource** vbo_resource,
		unsigned short* depthRaw, unsigned char* colorRaw,
		unsigned int w, unsigned int h,
		float fx, float fy, float ppx, float ppy, float depthScale, float depthThreshold
	);
	static void depthMapTriangulate(
		struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
		unsigned int w, unsigned int h,int *count,float degree
	);
};