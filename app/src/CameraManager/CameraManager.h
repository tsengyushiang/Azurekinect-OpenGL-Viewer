#pragma once

#include "./CameraGL.h"
#include "../InputCamera/Realsense.h"
#include "../InputCamera/JsonData.h"
#include "../InputCamera/AzureKinect.h"
#include <functional>

#define CamIterator std::vector<CameraGL>::iterator&
#define RecordContainer FILE
class CameraManager {

	rs2::context ctx;
	std::vector<CameraGL> cameras;

	void addJsonDevice(std::string serial, std::string filePath);
	void addRealsense(std::string serial, int cw, int ch, int dw, int wh);
	void addAzuekinect(int index);
	void removeDevice(CamIterator device);

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

	size_t size();

	void recordFrame();

	int debugInputDeviceIndex = -1;
	void getInputDebugDevice(std::function<void(CameraGL)> callback);
	
	int debugOutputDeviceIndex = -1;
	void getOutputDebugDevice(std::function<void(CameraGL)> callback);

	void getAllDevice(std::function<void(CamIterator)> callback);
	void getAllDevice(std::function<void(CamIterator, std::vector<CameraGL>&)> callback);
	int getProjectTextureDevice(std::function<void(CamIterator)> callback);
	void getFoward3DWrappingDevice(std::function<void(CamIterator)> callback);

	void updateProjectTextureWeight(glm::mat4 vmodelMat);
};