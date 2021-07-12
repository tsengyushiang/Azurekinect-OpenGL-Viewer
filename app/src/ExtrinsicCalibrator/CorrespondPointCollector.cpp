#include "./CorrespondPointCollector.h"

CorrespondPointCollector::CorrespondPointCollector(InputBase* srcCam, InputBase* trgCam, int count, float threshold) {
	sourcecam = srcCam;
	targetcam = trgCam;
	size = count;
	pushThresholdmin = threshold;
	source = (float*)calloc(size * 3 * 2, sizeof(float));
	target = (float*)calloc(size * 3 * 2, sizeof(float));
	result = (float*)calloc(size * 3 * 2, sizeof(float));
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	//sourcecam->opencvImshow = true;
	//targetcam->opencvImshow = true;
}
CorrespondPointCollector:: ~CorrespondPointCollector() {
	free(source);
	free(target);
	free(result);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);

	//sourcecam->opencvImshow = false;
	//targetcam->opencvImshow = false;
}

void CorrespondPointCollector::render(glm::mat4 mvp, GLuint shader_program) {
	ImguiOpeGL3App::setPointsVAO(vao, vbo, source, size);
	ImguiOpeGL3App::render(mvp, 10, shader_program, vao, size, GL_POINTS);

	ImguiOpeGL3App::setPointsVAO(vao, vbo, target, size);
	ImguiOpeGL3App::render(mvp, 10, shader_program, vao, size, GL_POINTS);
}

bool CorrespondPointCollector::pushCorrepondPoint(glm::vec3 src, glm::vec3 trg) {

	int index = vaildCount;
	if (index != 0) {
		// check threshold
		glm::vec3 p;
		int previousIndex = index - 1;

		for (auto p : srcPoint) {
			auto d = glm::length(p - src);
			if (d < pushThresholdmin) {
				return false;
			}
		}

		for (auto p : dstPoint) {
			auto d = glm::length(p - trg);
			if (d < pushThresholdmin) {
				return false;
			}
		}
	}

	srcPoint.push_back(src);
	dstPoint.push_back(trg);

	source[index * 6 + 0] = src.x;
	source[index * 6 + 1] = src.y;
	source[index * 6 + 2] = src.z;
	source[index * 6 + 3] = 1.0;
	source[index * 6 + 4] = 0.0;
	source[index * 6 + 5] = 0.0;

	target[index * 6 + 0] = trg.x;
	target[index * 6 + 1] = trg.y;
	target[index * 6 + 2] = trg.z;
	target[index * 6 + 3] = 0.0;
	target[index * 6 + 4] = 1.0;
	target[index * 6 + 5] = 0.0;

	vaildCount++;

	return true;
}

void CorrespondPointCollector::calibrate() {
	glm::mat4 transform = pcl_pointset_rigid_calibrate(size, srcPoint, dstPoint);
	sourcecam->modelMat = transform * sourcecam->modelMat;
	sourcecam->calibrated = true;
}