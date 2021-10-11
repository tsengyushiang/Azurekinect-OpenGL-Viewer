#pragma once

#include<iostream>
#include <functional>
#include <GL/gl3w.h>            // Initialize with gl3wInit()
#include <GLFW/glfw3.h>         // Include glfw3.h after our OpenGL definitions
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../config.h"

class CudaOpenGL {
public:
	static void createBufferObject(GLuint* vbo, struct cudaGraphicsResource** vbo_res,unsigned int vbo_res_flags, unsigned int size, int type);
	static void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);
	static void bindReadOnlyGLTextureforCuda(GLuint* tex_screen, struct cudaGraphicsResource* cuda_tex_screen_resource);
	static void createCudaGLTexture(GLuint* textureID, cudaGraphicsResource_t* cudaResources, int w, int h);
	static void deleteCudaGLTexture(GLuint* textureID, cudaGraphicsResource_t* cudaResources);
};

class CudaGLDepth2PlaneMesh {
public :
	//cuda opengl
	int* cudaIndicesCount = 0;
	uint16_t* cudaDepthData = 0;
	uint16_t* cudaDilatedDepthData = 0;
	unsigned char* cudaColorData = 0;
	GLuint vao, vbo, ibo;
	struct cudaGraphicsResource* cuda_vbo_resource, * cuda_ibo_resource;
	int width = 1280;
	int height = 720;

	CudaGLDepth2PlaneMesh(int w, int h,int colorchannel);
	void destory();
	void render(std::function<void(GLuint& vao, int& count)>callback);

	void getMeshData(float** posnormalvertexArr, unsigned int** faceIndices,int* faceCount);
};