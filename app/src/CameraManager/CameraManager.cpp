#include "./CameraManager.h"

namespace ImGui
{
	static auto vector_getter = [](void* vec, int idx, const char** out_text)
	{
		auto& vector = *static_cast<std::vector<std::string>*>(vec);
		if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
		*out_text = vector.at(idx).c_str();
		return true;
	};

	bool ComboVec(const char* label, int* currIndex, std::vector<std::string>& values)
	{
		if (values.empty()) { return false; }
		return Combo(label, currIndex, vector_getter,
			static_cast<void*>(&values), values.size());
	}

	bool ListBoxVec(const char* label, int* currIndex, std::vector<std::string>& values)
	{
		if (values.empty()) { return false; }
		return ListBox(label, currIndex, vector_getter,
			static_cast<void*>(&values), values.size());
	}

}

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
	device = cameras.erase(device);
	device->destory();
}

void CameraManager::setExtrinsicsUI() {
	if (ImGui::Button("load camera pose"))
		ImGuiFileDialog::Instance()->OpenDialog("load camera pose", "load camera pose", ".json", ".");
	if (ImGuiFileDialog::Instance()->Display("load camera pose"))
	{
		// action if OK
		if (ImGuiFileDialog::Instance()->IsOk())
		{
			std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
			std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
			// action
			std::map<std::string, glm::mat4> poses;
			std::vector<Jsonformat::CamPose> setting;
			JsonUtils::loadCameraPoses(filePathName, setting);
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
		// close
		ImGuiFileDialog::Instance()->Close();
	}
	ImGui::SameLine();
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
		
	ImGui::Text("ShowOutput : ");
	for (int i = 0; i < cameras.size(); i++) {
		ImGui::RadioButton(KEY("showOutput", cameras[i].camera), &debugOutputDeviceIndex, i);
	}
	ImGui::RadioButton("None##showOutput", &debugOutputDeviceIndex, -1);

	ImGui::Text("SaveOutput : ");
	for (int i = 0; i < cameras.size(); i++) {
		if(ImGui::Button(KEY("SaveOutput", cameras[i].camera))){
			cameras[i].saveWrappedResult();
		}
	}

	ImGui::Text("UseDevice : ");
	for (auto device = cameras.begin(); device != cameras.end(); device++) {
		ImGui::Checkbox(KEY("useDepth", device->camera), &(device->useDepth));
	}
}

void CameraManager::recordFrame() {

	const int MAXCAPTUREFRAME = 5000;
	if (cameras.size() > 0 && recording) {
		
		if(recordFrameCount==0){
			for (auto cam : cameras) {
				cam.camera->startRecord(MAXCAPTUREFRAME);
				/*
				//one file one cam
				record_color_frames.push_back(fopen(std::string(cam.camera->serial + "color.bin").c_str(), "wb"));
				record_depth_frames.push_back(fopen(std::string(cam.camera->serial + "depth.bin").c_str(), "wb"));
				*/
			}
		}

		for (int i = 0; i < cameras.size();i++) {
			auto cam = cameras[i].camera;

			int width = cam->width;
			int height = cam->height;
			
			int maxCache = cam->MAXCACHE;
			//record to cache
			memcpy((void*)cam->color_cache[recordFrameCount%maxCache], cam->p_color_frame, INPUT_COLOR_CHANNEL * width * height * sizeof(uchar));
			memcpy((void*)cam->depth_cache[recordFrameCount%maxCache], cam->p_depth_frame, width * height * sizeof(uint16_t));
			cam->curRecordFrame++;
			/*
			fwrite(
				cam->p_color_frame,
				INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char), 
				1,
				record_color_frames[i]
			);
			fflush(record_color_frames[i]);
			fwrite(
				cam->p_depth_frame, 
				width * height * sizeof(uint16_t), 
				1,
				record_depth_frames[i]
			);
			fflush(record_depth_frames[i]);
			*/

		}

		if (recordFrameCount+1 < MAXCAPTUREFRAME) {
			recordFrameCount++;
		}
		/*
		if (recordFrameCount >= MAXCAPTUREFRAME) {
			recordFrameCount = 0;
			recording = false;
			for (int i = 0; i < cameras.size(); i++) {
				fclose(record_color_frames[i]);
				fclose(record_depth_frames[i]);
				record_color_frames.clear();
				record_depth_frames.clear();
			}
		}
		*/
	}
}

void CameraManager::addLocalFileUI() {
	if (ImGui::Button("Add Camera Frame(s)"))
		ImGuiFileDialog::Instance()->OpenDialog("Add Camera Frame(s)", "Add Camera Frame(s)", ".json", ".", 0);
	if (ImGuiFileDialog::Instance()->Display("Add Camera Frame(s)"))
	{
		// action if OK
		if (ImGuiFileDialog::Instance()->IsOk())
		{
			auto filePathMap = ImGuiFileDialog::Instance()->GetSelection();
			for (auto nameAndPath : filePathMap) {
				jsonDataLen = addJsonDevice(nameAndPath.first.substr(0, nameAndPath.first.size() - 5), nameAndPath.second);
			}
		}
		// close
		ImGuiFileDialog::Instance()->Close();
	}

	// time control
	{
		std::vector<std::string> index;
		for (int i = 0; i < jsonDataLen; i++) {
			index.push_back(std::to_string(i));
		}
		ImGui::ListBoxVec("Time", &time, index);
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			device->camera->syncTime = time;
		}
	}
}

void CameraManager::addCameraUI() {
		
	camRecroder.addGUI(cameras);

	if (ImGui::Button("Toggle create point cloud or not")) {
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			device->create3d = !device->create3d;
		}
	}
	if (ImGui::Button(std::string("Already Record "+std::to_string(recordFrameCount)+" frames").c_str())) {
		recording = !recording;
	}

	if (ImGui::Button("Refresh")) {
		Realsense::updateAvailableSerialnums(ctx);
		AzureKinect::updateAvailableSerialnums();
	}
	ImGui::Text("AzureKinect device :");
	ImGui::RadioButton("Master##azurekinect", &AzureKinect::ui_isMaster, 1);
	ImGui::SameLine();
	ImGui::RadioButton("Sub##azurekinect", &AzureKinect::ui_isMaster, 0);
	for (int i = 0; i < AzureKinect::availableSerialnums.size(); i++) {
		if (ImGui::Button((AzureKinect::availableSerialnums[i]+"##azurekinect").c_str())) {
			addAzuekinect(i);
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

int CameraManager::addJsonDevice(std::string serial, std::string filePath) {

	int w, h;
	JsonUtils::loadResolution(filePath, w, h);
	
	JsonData* camera = new JsonData(w,h);
	camera->serial = serial;
	
	if (!JsonUtils::loadSinglCamTimeSequence(filePath, camera->framefiles)){
		camera->framefiles.push_back(filePath);
	}

	CameraGL device(camera);

	cameras.push_back(device);

	return camera->framefiles.size();
}

void CameraManager::addAzuekinect(int index) {
	AzureKinect* azureKinect = new AzureKinect();
	azureKinect->runDevice(index,AzureKinect::ui_isMaster);
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

void CameraManager::getOutputDebugDevice(std::function<void(CameraGL)> callback) {
	if (debugOutputDeviceIndex < 0 || debugOutputDeviceIndex >= cameras.size())return;
	callback(cameras[debugOutputDeviceIndex]);
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

void CameraManager::getFoward3DWrappingDevice(std::function<void(CamIterator)> callback) {
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

