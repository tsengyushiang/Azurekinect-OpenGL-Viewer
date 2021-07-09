#pragma once

#include "../ImguiOpenGL/ImguiOpeGL3App.h"
#include "../cuda/CudaOpenGLUtils.h"
#include "../cuda/cudaUtils.cuh"
#include "../json/jsonUtils.h"

#include <glm/glm.hpp>
#include <opencv2\core\hal\interface.h>

class VirtualCam {
public:
	int w, h;
	float ppx, ppy, fx, fy;

	float distance = 0.616;
	float PolarAngle = 1.57;
	float AzimuthAngle = 4.732;
	float farplane = 1.5;
	float nearplane = 0;

	float distancemin = 0;
	float distanceMax = 5;
	float PolarAngleMax = 3.0;
	float PolarAnglemin = 0.1;
	float AzimuthAngleMax = 6.28;
	float AzimuthAnglemin = -6.28;

	glm::vec3 color = glm::vec3(1.0, 1.0, 0.0);

	std::vector<glm::mat4> modelMats;

	GLFrameBuffer viewport;
	GLFrameBuffer debugview;

	bool isFromFile = true;

	uchar* colorRaw;
	float* depthRaw;
	uint16_t* depthintRaw;

	glm::mat4 getModelMat(glm::vec3 lookAtPoint, int curFrame);

	VirtualCam(int width, int height);

	void save(std::string filename, glm::vec3 lookAtPoint, int curFrame);
	void addUI();

	void renderFrustum(
		glm::mat4 deviceMVP,
		GLuint& vao, GLuint& vbo, GLuint render_vertexColor_program);
};