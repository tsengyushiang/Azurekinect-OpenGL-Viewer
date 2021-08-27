#pragma once

#include "./CameraGL.h"
#include "./CameraRecorder.h"
#include "../InputCamera/Realsense.h"
#include "../InputCamera/JsonData.h"
#include "../InputCamera/AzureKinect.h"
#include <functional>
#include <thread>

using namespace std;

#define RecordContainer FILE

class CameraManager {

	rs2::context ctx;
	std::vector<CameraGL> cameras;
	CameraRecorder camRecroder;

	// return length of file
	int jsonDataLen = 0;
	int addJsonDevice(std::string serial, std::string filePath);
	
	void addRealsense(std::string serial, int cw, int ch, int dw, int wh);
	void addAzuekinect(int index);
	void removeDevice(CamIterator device);

	int time = 0;

public :
	std::vector<RecordContainer*>record_depth_frames;
	std::vector<RecordContainer*>record_color_frames;
	 
	int recordFrameCount = 0;
	bool recording = false;

	CameraManager();
	void destory();
	void deleteIdleCam();

	void addCameraUI();
	void setExtrinsicsUI();
	void addDepthAndTextureControlsUI();
	void addLocalFileUI();

	size_t size();

	void recordFrame();
	
	int debugOutputDeviceIndex = -1;
	void getOutputDebugDevice(std::function<void(CameraGL)> callback);

	void getAllDevice(std::function<void(CamIterator)> callback);
	void getAllDevice(std::function<void(CamIterator, std::vector<CameraGL>&)> callback);
	int getProjectTextureDevice(std::function<void(CamIterator)> callback);
	void getFoward3DWrappingDevice(std::function<void(CamIterator)> callback);

	void updateProjectTextureWeight(glm::mat4 vmodelMat);
};