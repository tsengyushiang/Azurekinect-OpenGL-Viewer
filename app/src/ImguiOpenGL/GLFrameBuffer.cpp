#include "ImguiOpeGL3App.h"

void GLFrameBuffer::createFrameBuffer(
    GLuint* framebuffer,
    GLuint* texColorBuffer, GLuint* depthBuffer, GLuint* rbo,
    int w, int h
) {
    glGenFramebuffers(1, framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, *framebuffer);

    glGenTextures(1, texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, *texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *texColorBuffer, 0);

    glGenTextures(1, depthBuffer);
    glBindTexture(GL_TEXTURE_2D, *depthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *depthBuffer, 0);

    //glGenRenderbuffers(1, rbo);
    //glBindRenderbuffer(GL_RENDERBUFFER, *rbo);
    //glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    //glBindRenderbuffer(GL_RENDERBUFFER, 0);

    //glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, *rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLFrameBuffer::GLFrameBuffer(int w,int h):
    width(w),
    height(h) 
{
    GLFrameBuffer::createFrameBuffer(&framebuffer, &texColorBuffer, &depthBuffer, &rbo, w, h);
}

void GLFrameBuffer::render(std::function<void()> callback) {

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_FRONT);

	callback();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

unsigned char* GLFrameBuffer::getRawColorData() {
    unsigned char* colorRaw = new unsigned char[width * height * 4];
	glBindTexture(GL_TEXTURE_2D, texColorBuffer);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, colorRaw);
    return colorRaw;
}