#include "CudaOpenGLUtils.h"

void CudaOpenGL::bindReadOnlyGLTextureforCuda(GLuint* tex_screen, struct cudaGraphicsResource* cuda_tex_screen_resource) {
    cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *tex_screen, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
}

void CudaOpenGL::createBufferObject(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags, unsigned int size,int type)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(type, *vbo);

    // initialize buffer object
    // example : unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(type, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(type, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void CudaOpenGL::deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{
    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}