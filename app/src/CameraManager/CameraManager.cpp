#include "./CameraManager.h"

CameraManager::CameraManager() {
	Realsense::updateAvailableSerialnums(ctx);
	AzureKinect::updateAvailableSerialnums();
}

void CameraManager::destory() {
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		removeDevice(device);
	}
}

void CameraManager::removeDevice(CamIterator device) {
	Realsense::updateAvailableSerialnums(ctx);
	AzureKinect::updateAvailableSerialnums();
	device = cameras.erase(device);
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

void CameraManager::addDepthAndTextureControlsUI() {

	auto KEY = [this](std::string keyword,InputBase* camera)->const char* {
		return (camera->serial + std::string("##") + keyword).c_str();
	};

	ImGui::Text("ShowInput : ");
	for (int i = 0; i < cameras.size(); i++) {
		ImGui::RadioButton(KEY("showInput", cameras[i].camera), &debugDeviceIndex, i);
	}
	ImGui::RadioButton("None##showInput", &debugDeviceIndex, -1);

	ImGui::Text("UseDepth : ");
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		ImGui::Checkbox(KEY("useDepth", device->camera), &(device->useDepth));
	}
	ImGui::Text("UseTexture : ");
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		ImGui::Checkbox(KEY("useTexture", device->camera), &(device->useTexture));
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

	// list all usb3.0 realsense device
	if (ImGui::Button("Refresh")) {
		Realsense::updateAvailableSerialnums(ctx);
		AzureKinect::updateAvailableSerialnums();
	}
	ImGui::SameLine();
	if (ImGui::Button("snapshot all")) {
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			device->save();
		}
	}

	ImGui::Text("AzureKinect device :");
	//waiting Azure kinect activated device
	for (std::string serial : AzureKinect::availableSerialnums) {
		bool alreadyStart = false;
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			if (device->camera->serial == serial) {
				alreadyStart = true;
				break;
			}
		}
		if (!alreadyStart)
		{
			ImGui::Text(serial.c_str());

			if (ImGui::Button(("1080p##Azruekinect" + serial).c_str())) {
				addAzuekinect(serial, 1920, 1080);
			}
			ImGui::SameLine();
			if (ImGui::Button(("720p##Azruekinect" + serial).c_str())) {
				addAzuekinect(serial, 1280, 720);
			}

		}
	}

	ImGui::Text("Realsense device :");
	// waiting realsense active device
	for (std::string serial : Realsense::availableSerialnums) {
		bool alreadyStart = false;
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			if (device->camera->serial == serial) {
				alreadyStart = true;
				break;
			}
		}
		if (!alreadyStart)
		{
			ImGui::Text(serial.c_str());

			if (ImGui::Button(("c1920d1280##"+ serial).c_str())) {
				addRealsense(serial,1920,1080,1280,720);
			}
			ImGui::SameLine();
			if (ImGui::Button(("c1280d1280##" + serial).c_str())) {
				addRealsense(serial, 1280, 720, 1280, 720);
			}
			ImGui::SameLine();
			// for L515
			if (ImGui::Button(("c1920d1024##" + serial).c_str())) {
				addRealsense(serial, 1920, 1080, 1024, 768);
			}
			ImGui::SameLine();
			if (ImGui::Button(("c1280d1024##" + serial).c_str())) {
				addRealsense(serial, 1280, 720, 1024, 768);
			}
		}
	}
	

	// Running device
	ImGui::Text("Running Device :");
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		device->addui();
	}	
}

void CameraManager::addJsonDevice(std::string serial) {

	int w, h;
	JsonUtils::loadRealsenseJson(serial, w, h);
	
	JsonData* camera = new JsonData(w,h);
	camera->serial = serial;
	
	JsonUtils::loadRealsenseJson(serial,
		camera->width,
		camera->height,
		camera->intri.fx,
		camera->intri.fy,
		camera->intri.ppx,
		camera->intri.ppy,
		camera->frameLength,
		camera->intri.depth_scale,
		&camera->p_depth_frame,
		&camera->p_color_frame);

	CameraGL device(camera);

	cameras.push_back(device);
}

void CameraManager::addAzuekinect(std::string serial, int cw, int ch) {
	AzureKinect* azureKinect = new AzureKinect(cw,ch);
	azureKinect->runDevice(serial);
	CameraGL device(azureKinect);
	cameras.push_back(device);
}

void CameraManager::addRealsense(std::string serial,int cw,int ch,int dw,int dh) {
	try {
		Realsense* realsense = new Realsense(cw, ch, dw, dh);
		realsense->runDevice(serial.c_str(), ctx);
		CameraGL device(realsense);
		cameras.push_back(device);
	}
	catch (...) {
		std::cout << "Add device { " << serial << " } Error: use offical viewer check your deivce first." << std::endl;
	}
}

void CameraManager::getSingleDebugDevice(std::function<void(CameraGL)> callback) {
	if (debugDeviceIndex < 0 || debugDeviceIndex >= cameras.size())return;
	callback(cameras[debugDeviceIndex]);
}

void CameraManager::getAllDevice(std::function<void(CamIterator)> callback) {
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		callback(device);
	}
}
void CameraManager::getAllDevice(std::function<void(CamIterator, std::vector<CameraGL>& allDevice)> callback) {
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		callback(device, cameras);
	}
}

int CameraManager::getProjectTextureDevice(std::function<void(CamIterator)> callback) {
	int count = 0;
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		if (device->useTexture) {
			callback(device);
			count++;
		}
	}
	return count;
}

void CameraManager::getFowardDepthWarppingDevice(std::function<void(CamIterator)> callback) {
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		if (device->useDepth) {
			callback(device);
		}
	}
}

size_t CameraManager::size() {
	return cameras.size();
}

void CameraManager::updateProjectTextureWeight(glm::mat4 vmodelMat) {
	glm::vec3 virtualviewPosition = glm::vec3(
		vmodelMat[3][0],
		vmodelMat[3][1],
		vmodelMat[3][2]
	);

	int nearestDeviceIndex = -1;
	float distance = 99999;
	for (int i = 0; i < cameras.size(); i++) {
		auto device = cameras[i];
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
	for (int i = 0; i < cameras.size(); i++) {
		cameras[i].weight = (i == nearestDeviceIndex) ? 99 : 1;
	}
}

void CameraManager::deleteIdleCam() {
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		if (device->ready2Delete) {
			removeDevice(device);
			device--;
			continue;
		}
	}
}

