#pragma once

#include "./CorrespondPointCollector.h"
#include "../opencv/opecv-utils.h"
#include "../CameraManager/CameraGL.h"
#include "../eigen/EigenUtils.h"
#include "../pcl/examples-pcl.h"

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

	float icp_correspondThreshold = 0.05;

public :

	void addUI();
	void render(glm::mat4 mvp, GLuint shader_program);

	int uniformDepthSample = 2;
	glm::mat4 runIcp(InputBase* pcdSource, InputBase* pcdTarget,int step);
	void calibrateCollectedPoints(bool reset = false);
	// detect aruco to calibrate unregisted camera
	void waitCalibrateCamera(std::vector<CameraGL>::iterator device, std::vector<CameraGL>& allDevice);
	void collectCalibratePoints();
	void alignDevice2calibratedDevice(InputBase* uncalibratedCam, std::vector<CameraGL>& allDevice);
	void fitMarkersOnEstimatePlane(InputBase* camera,
		std::function<std::vector<glm::vec2>(int, int, uchar*, int)> findMarkerPoints);
	CalibrateResult putAruco2Origion(InputBase* camera);
	bool checkIsCalibrating(std::string serial, glm::vec3& index);
};