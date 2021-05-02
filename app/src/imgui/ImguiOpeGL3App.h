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
	void initImguiOpenGL3(float width = 1280, float height = 720);
	virtual void initGL();
	virtual void mainloop();
	virtual void mousedrag(float,float);
	static void glfw_error_callback(int error, const char* description);

	// opengl camera
	void setcamera(float width, float height);
	float fov = 60;
	float distance = 3;
	float PolarAngle = 0;
	float AzimuthAngle = 0;
	glm::mat4 mvp;

	//---------------------following method implement in ImguiOpeGL3App_gl.cpp

	// opengl functions
	static bool check_shader_compile_status(GLuint obj);
	static bool check_program_link_status(GLuint obj);

	// opengl render pointcloud 
	static GLuint genPointcloudShader(GLFWwindow* window);
	static void setPointsVAO(GLuint& vao, GLuint& vbo,GLfloat* vertexData, float size);
	static void renderPoints(glm::mat4& mvp, float pointsize,GLuint shader_program, GLuint vao, float size);
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