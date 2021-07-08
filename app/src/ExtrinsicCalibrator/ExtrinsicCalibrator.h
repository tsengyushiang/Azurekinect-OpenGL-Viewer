#pragma once

#include "./CorrespondPointCollector.h"
#include "../opencv/opecv-utils.h"
#include "../CameraManager/CameraGL.h"

typedef struct calibrateResult {
	std::vector<glm::vec3> points;
	glm::mat4 calibMat;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	bool success;
}CalibrateResult;

class ExtrinsicCalibrator {

	CorrespondPointCollector* calibrator = nullptr;
	int collectPointCout = 15;
	float collectthreshold = 0.1f;

public :

	void addUI();
	void render(glm::mat4 mvp, GLuint shader_program);

	// detect aruco to calibrate unregisted camera
	void waitCalibrateCamera(std::vector<CameraGL>::iterator device, std::vector<CameraGL>& allDevice);
	void collectCalibratePoints();
	void alignDevice2calibratedDevice(RealsenseDevice* uncalibratedCam, std::vector<CameraGL>& allDevice);
	CalibrateResult putAruco2Origion(RealsenseDevice* camera);
	bool checkIsCalibrating(std::string serial, glm::vec3& index);
};