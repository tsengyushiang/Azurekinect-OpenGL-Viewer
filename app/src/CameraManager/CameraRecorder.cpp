#include "./CameraRecorder.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <filesystem>
#include <iomanip>
#include <ctime>

void CameraRecorder::addGUI(std::vector<CameraGL> cameras){

	if (ImGui::Button(("Snapshot all, already buffered :" + std::to_string(bufferFramesCount)).c_str())) {
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			uint16_t* p_depth_frame = (uint16_t*)calloc(device->camera->width * device->camera->height, sizeof(uint16_t));
			memcpy((void*)p_depth_frame, (uint16_t*)(void*)device->camera->p_depth_frame, device->camera->width * device->camera->height * sizeof(uint16_t));
			
			bufferFramesCount++;
			getBuffer(device->camera->serial)->push_back({
				device->camera,
				device->getProcessedColorFrame(),
				p_depth_frame
				});
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("export")) {
		exportBuffer2files();
	}
}
void CameraRecorder::exportBuffer2files() {

	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream sstr;
	sstr <<"./"<<std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
	std::string folder= sstr.str();
	cv::utils::fs::createDirectory(folder);

	std::vector<std::string> camserials;
	for (auto buffer : buffers) {
		auto frames = *buffer.second;

		std::vector<std::string> camfilenames;

		for (auto data : frames) {
			auto camera = data.camera;

			// save every camera's data
			std::ostringstream stringStream;
			stringStream <<"./"<<camera->serial <<"_t"<<std::setw(8) << std::setfill('0') << camfilenames.size();
			std::string filename = cv::utils::fs::join(folder, stringStream.str());
			camfilenames.push_back(stringStream.str()+".json");

			JsonUtils::saveRealsenseJson(
				filename,
				camera->width, camera->height,
				camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
				camera->intri.depth_scale, data.p_depth_frame, data.colorRaw,
				camera->farPlane
			);
			free(data.colorRaw);
			free(data.p_depth_frame);
		}

		camserials.push_back("./"+buffer.first+".json");
		JsonUtils::saveSingleCamTimeInfo(
			cv::utils::fs::join(folder, buffer.first),
			camfilenames
		);

		frames.clear();
		bufferFramesCount = 0;
	}

	JsonUtils::saveMultiCamSyncInfo(
		cv::utils::fs::join(folder, sstr.str()),
		camserials
	);

	for (auto buffer : buffers) {
		free(buffer.second);
	}
	buffers.clear();
}
std::vector<MannulBufferingFrame>* CameraRecorder::getBuffer(std::string serialnum) {
	if (buffers.find(serialnum) == buffers.end()) {
		buffers[serialnum] = new std::vector<MannulBufferingFrame>();
	}
	return buffers[serialnum];
}

CameraRecorder::~CameraRecorder() {
	for (auto buffer : buffers) {
		free(buffer.second);
	}
}

