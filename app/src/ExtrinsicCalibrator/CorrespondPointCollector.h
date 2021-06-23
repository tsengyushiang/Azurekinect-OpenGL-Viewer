#pragma once
#include "../imgui/ImguiOpeGL3App.h"
#include "../realsnese//RealsenseDevice.h"
#include "../pcl/examples-pcl.h"

class CorrespondPointCollector {

public:
	GLuint vao, vbo;

	RealsenseDevice* sourcecam;
	RealsenseDevice* targetcam;
	int vaildCount = 0;
	int size;
	float pushThresholdmin = 0.1f;

	std::vector<glm::vec3> srcPoint;
	std::vector<glm::vec3> dstPoint;
	float* source;
	float* target;
	float* result;
	CorrespondPointCollector(RealsenseDevice* srcCam, RealsenseDevice* trgCam, int count = 10, float threshold = 0.2f);		
	~CorrespondPointCollector();

	void render(glm::mat4 mvp, GLuint shader_program);
	bool pushCorrepondPoint(glm::vec3 src, glm::vec3 trg);
	void calibrate();
};