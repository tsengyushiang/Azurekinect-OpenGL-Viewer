#include "./CameraManager.h"

CameraManager::CameraManager() {
	serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
}

void CameraManager::destory() {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		removeDevice(device);
	}
}

void CameraManager::removeDevice(CamIterator device) {
	delete device->camera;
	serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
	device = realsenses.erase(device);
	device->destory();
}

void CameraManager::setExtrinsicsUI() {
	static char jsonfilename[100] = "CameraExtrinsics";
	ImGui::Text("jsonfilename: ");

	ImGui::SameLine();
	if (ImGui::Button("load camera pose")) {
		std::map<std::string, glm::mat4> poses;
		std::vector<Jsonformat::CamPose> setting;
		JsonUtils::loadCameraPoses(jsonfilename, setting);
		for (auto cam : setting) {
			poses[cam.id] = glm::mat4(
				cam.extrinsic[0], cam.extrinsic[4], cam.extrinsic[8], cam.extrinsic[12],
				cam.extrinsic[1], cam.extrinsic[5], cam.extrinsic[9], cam.extrinsic[13],
				cam.extrinsic[2], cam.extrinsic[6], cam.extrinsic[10], cam.extrinsic[14],
				cam.extrinsic[3], cam.extrinsic[7], cam.extrinsic[11], cam.extrinsic[15]
			);
			std::cout << cam.extrinsic[0] << ", " << cam.extrinsic[4] << "," << cam.extrinsic[8] << "," << cam.extrinsic[12] << std::endl;
			std::cout << cam.extrinsic[1] << ", " << cam.extrinsic[5] << "," << cam.extrinsic[9] << "," << cam.extrinsic[13] << std::endl;
			std::cout << cam.extrinsic[2] << ", " << cam.extrinsic[6] << "," << cam.extrinsic[10] << "," << cam.extrinsic[14] << std::endl;
			std::cout << cam.extrinsic[3] << ", " << cam.extrinsic[7] << "," << cam.extrinsic[11] << "," << cam.extrinsic[15] << std::endl;

		}

		getAllDevice([&poses](auto device) {
			device->camera->calibrated = true;
			device->camera->modelMat = poses[device->camera->serial];
		});
	}

	ImGui::SameLine();
	ImGui::InputText("##jsonfilenameurlInput", jsonfilename, 100);

	if (ImGui::Button("save camera pose")) {
		std::vector<Jsonformat::CamPose> setting;
		getAllDevice([&setting](auto device) {
			glm::mat4 modelMat = device->camera->modelMat;
			std::vector<float> extrinsic = {
				modelMat[0][0],modelMat[1][0],modelMat[2][0],modelMat[3][0],
				modelMat[0][1],modelMat[1][1],modelMat[2][1],modelMat[3][1],
				modelMat[0][2],modelMat[1][2],modelMat[2][2],modelMat[3][2],
				modelMat[0][3],modelMat[1][3],modelMat[2][3],modelMat[3][3]
			};
			Jsonformat::CamPose c = { device->camera->serial ,extrinsic };
			setting.push_back(c);
		});
		JsonUtils::saveCameraPoses(setting);
	}
}

void CameraManager::addCameraUI() {
	static char jsonfilename[20] = "932122060549";
	ImGui::Text("jsonfilename: ");
	ImGui::SameLine();
	if (ImGui::Button("add##jsonfilename")) {
		addJsonDevice(jsonfilename);
	}
	ImGui::SameLine();
	ImGui::InputText("##jsonfilenameurlInput", jsonfilename, 20);

	// input url for network device
	static char url[20] = "192.168.0.106";
	ImGui::Text("Network device Ip: ");
	ImGui::SameLine();
	if (ImGui::Button("add")) {
		addNetworkDevice(url);
	}
	ImGui::SameLine();
	ImGui::InputText("##urlInput", url, 20);

	// list all usb3.0 realsense device
	if (ImGui::Button("Refresh"))
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
	ImGui::SameLine();
	ImGui::Text("Realsense device :");

	// waiting active device
	for (std::string serial : serials) {
		bool alreadyStart = false;
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			if (device->camera->serial == serial) {
				alreadyStart = true;
				break;
			}
		}
		if (!alreadyStart)
		{
			if (ImGui::Button(serial.c_str())) {
				addDevice(serial);
			}
		}
	}
	{
		ImGui::Text("Debug :");
		if (ImGui::Button("snapshot all")) {
			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
				device->save();
			}
		}
	}
	// Running device
	ImGui::Text("Running Realsense device :");
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		device->addui();
	}	
}

// deprecated
void CameraManager::addNetworkDevice(std::string url) try {
	CameraGL device;

	device.camera = new RealsenseDevice();
	std::string serial = device.camera->runNetworkDevice(url, ctx);

	realsenses.push_back(device);
}
catch (const rs2::error& e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
}

void CameraManager::addJsonDevice(std::string serial) {
	CameraGL device;

	device.camera = new JsonRealsenseDevice();
	JsonUtils::loadRealsenseJson(serial,
		device.camera->width,
		device.camera->height,
		device.camera->intri.fx,
		device.camera->intri.fy,
		device.camera->intri.ppx,
		device.camera->intri.ppy,
		device.camera->frameLength,
		device.camera->intri.depth_scale,
		&device.camera->p_depth_frame,
		&device.camera->p_color_frame);
	device.camera->serial = serial;

	realsenses.push_back(device);
}

void CameraManager::addDevice(std::string serial) {
	CameraGL device;
	device.camera->runDevice(serial.c_str(), ctx);
	realsenses.push_back(device);
}

void CameraManager::getAllDevice(std::function<void(CamIterator)> callback) {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		callback(device);
	}
}
void CameraManager::getAllDevice(std::function<void(CamIterator, std::vector<CameraGL>& allDevice)> callback) {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		callback(device, realsenses);
	}
}

void CameraManager::getProjectTextureDevice(std::function<void(CamIterator)> callback) {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		callback(device);
	}
}

void CameraManager::getFowardDepthWarppingDevice(std::function<void(CamIterator)> callback) {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		callback(device);
	}
}

size_t CameraManager::size() {
	return realsenses.size();
}

void CameraManager::updateProjectTextureWeight(glm::mat4 vmodelMat) {
	glm::vec3 virtualviewPosition = glm::vec3(
		vmodelMat[3][0],
		vmodelMat[3][1],
		vmodelMat[3][2]
	);

	int nearestDeviceIndex = -1;
	float distance = 99999;
	for (int i = 0; i < realsenses.size(); i++) {
		auto device = realsenses[i];
		glm::vec3 devicePosition = glm::vec3(
			device.camera->modelMat[3][0],
			device.camera->modelMat[3][1],
			device.camera->modelMat[3][2]
		);
		float d = glm::distance(virtualviewPosition, devicePosition);
		if (d < distance) {
			distance = d;
			nearestDeviceIndex = i;
		}
	}
	for (int i = 0; i < realsenses.size(); i++) {
		realsenses[i].weight = (i == nearestDeviceIndex) ? 99 : 1;
	}
}

void CameraManager::deleteIdleCam() {
	for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
		if (device->ready2Delete) {
			removeDevice(device);
			device--;
			continue;
		}
	}
}

