#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/gl3w.h>            // Initialize with gl3wInit()
#include <GLFW/glfw3.h>         // Include glfw3.h after our OpenGL definitions

#include <iostream>
#include <string>
#include <vector>

class ImguiOpeGL3App
{

	static void glfw_error_callback(int error, const char* description);

public:
	void initImguiOpenGL3(float width = 1280, float height = 720);
	virtual void initGL();
	virtual void mainloop();
	virtual void mousedrag(float,float);
};

