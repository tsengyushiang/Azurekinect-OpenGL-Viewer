#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/gl3w.h>            // Initialize with gl3wInit()
#include <GLFW/glfw3.h>         // Include glfw3.h after our OpenGL definitions

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <string>
#include <vector>

class ImguiOpeGL3App
{
public:

	//---------------------following method implement in ImguiOpeGL3App.cpp

	GLFWwindow* window;
	void initImguiOpenGL3(int width = 1280, int height = 720);
	virtual void initGL();
	virtual void mainloop();
	virtual void addGui();
	virtual void mousedrag(float,float);
	static void glfw_error_callback(int error, const char* description);

	// opengl camera
	void setcamera(float width, float height);
	float fov = 60;
	float distance = 3;
	float PolarAngle = 1.57;
	float PolarAngleMax = 3.0;
	float PolarAnglemin = 0.1;
	float AzimuthAngle = 0.1;
	float AzimuthAngleMax = 6.28;
	float AzimuthAnglemin = 0;
	float autoRotateSpeed = 0.0;
	bool camVertical = true;

	float sensity = 5e-2;
	glm::vec3 lookAtPoint;
	glm::mat4 Projection;
	glm::mat4 View;
	glm::mat4 Model;

	//---------------------following method implement in ImguiOpeGL3App_gl.cpp

	// opengl functions
	static bool check_shader_compile_status(GLuint obj);
	static bool check_program_link_status(GLuint obj);

	// opengl render pointcloud 
	static GLuint genPointcloudShader(GLFWwindow* window);
	static void setTexture(GLuint& image,const unsigned char* vertexData, int width, int height);
	
	static void setTrianglesVAOIBO(GLuint& vao, GLuint& vbo, GLuint& ibo, GLfloat* vertexData, int vertexSize,unsigned int* indices,int indicesSize);
	static void renderElements(glm::mat4& mvp, float pointsize, GLuint shader_program, GLuint vao, int size);

	static void setPointsVAO(GLuint& vao, GLuint& vbo,GLfloat* vertexData, float size);
	static void render(glm::mat4& mvp, float pointsize,GLuint shader_program, GLuint vao, float size, int type);

	//---------------------following method implement in ImguiOpeGL3App_glhelper.cpp

	//helper shape
	static void genCameraHelper(GLuint& vao, GLuint& vbo, float width, float height, float ppx, float ppy, float fx, float fy, glm::ivec3 color, float size);
	static void genOrigionAxis(GLuint& vao, GLuint& vbo);

};

//custum rener and init demo
//class App :public ImguiOpeGL3App {
//
//public:
//	App() :ImguiOpeGL3App() {}
//
//	void initGL() override {
//		std::cout << "create shader, VAO" << std::endl;
//	}
//	void mainloop() override {
//		std::cout << "Render" << std::endl;
//	}
//};