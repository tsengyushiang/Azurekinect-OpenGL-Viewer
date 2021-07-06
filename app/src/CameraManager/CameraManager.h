#pragma once

#include "./CameraGL.h"
#include "../realsnese//RealsenseDevice.h"
#include "../realsnese/JsonRealsenseDevice.h"
#include <functional>

#define CamIterator std::vector<CameraGL>::iterator&

class CameraManager {

	rs2::context ctx;
	std::vector<CameraGL> realsenses;
	std::set<std::string> serials;

	void addJsonDevice(std::string serial);
	void addDevice(std::string serial, int cw, int ch, int dw, int wh);
	void removeDevice(CamIterator device);

public :

	CameraManager();
	void destory();
	void deleteIdleCam();

	void addCameraUI();
	void setExtrinsicsUI();
	void addDepthAndTextureControlsUI();

	size_t size();
	void getAllDevice(std::function<void(CamIterator)> callback);
	void getAllDevice(std::function<void(CamIterator, std::vector<CameraGL>&)> callback);
	int getProjectTextureDevice(std::function<void(CamIterator)> callback);
	void getFowardDepthWarppingDevice(std::function<void(CamIterator)> callback);

	void updateProjectTextureWeight(glm::mat4 vmodelMat);
};