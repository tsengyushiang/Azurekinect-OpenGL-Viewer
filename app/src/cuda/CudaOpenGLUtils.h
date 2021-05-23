#pragma once

#include <GL/gl3w.h>            // Initialize with gl3wInit()
#include <GLFW/glfw3.h>         // Include glfw3.h after our OpenGL definitions

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class CudaOpenGL {
public:
	static void createBufferObject(GLuint* vbo, struct cudaGraphicsResource** vbo_res,unsigned int vbo_res_flags, unsigned int size, int type);
	static void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);
};