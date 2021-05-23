#include <cuda_runtime.h>

class CudaAlogrithm {
public:
	static void depthMap2point(struct cudaGraphicsResource** vbo_resource, unsigned int w, unsigned int h, float depthScale);
	static void depthMapTriangulate(
		struct cudaGraphicsResource** vbo_resource, struct cudaGraphicsResource** ibo_resource,
		unsigned int w, unsigned int h,int *count);
};