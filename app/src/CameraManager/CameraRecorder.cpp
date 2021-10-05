#include "./CameraRecorder.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <filesystem>
#include <iomanip>
#include <ctime>

void CameraRecorder::addGUI(std::vector<CameraGL> cameras) {
	auto KEY = [this](std::string keyword, std::string invisibleTag)->const char* {
		return (keyword + std::string("##") + invisibleTag).c_str();
	};

	ImGui::Text((folder + ", " + std::to_string(cameras.size())).c_str());
	ImGui::Text(("Already buffered " + std::to_string(getSize()) + STATE_MSG[state]).c_str());

	if (state == INMEMORY) {
		if (recordingMode) {
			if (ImGui::Button(KEY("Auto Record STOP", folder))) {
				recordingMode = false;
			}
		}
		else {
			if (ImGui::Button(KEY("Auto Record START", folder))) {
				recordingMode = true;
			}
		}
		ImGui::SameLine();
		if (recordingMode || ImGui::Button(KEY("Snapshot all", folder))) {
			for (auto device = cameras.begin(); device != cameras.end(); device++) {
				device->camera->enableUpdateFrame = false;
				uchar* p_color_frame = (unsigned char*)calloc(INPUT_COLOR_CHANNEL * device->camera->width * device->camera->height, sizeof(unsigned char));
				memcpy((void*)p_color_frame, (uint16_t*)(void*)device->camera->p_color_frame, INPUT_COLOR_CHANNEL * device->camera->width * device->camera->height * sizeof(uchar));

				uint16_t* p_depth_frame = (uint16_t*)calloc(device->camera->width * device->camera->height, sizeof(uint16_t));
				memcpy((void*)p_depth_frame, (uint16_t*)(void*)device->camera->p_depth_frame, device->camera->width * device->camera->height * sizeof(uint16_t));

				float* xy_table = (float*)calloc(device->camera->width * device->camera->height * 2, sizeof(float));
				memcpy((void*)xy_table, (uint16_t*)(void*)device->camera->xy_table, device->camera->width * device->camera->height * 2 * sizeof(float));
				device->camera->enableUpdateFrame = true;
				getBuffer(device->camera->serial)->push_back({
					device->camera,
					p_color_frame,
					p_depth_frame,
					xy_table
					});
			}
		}

		// copy memory data to camera to view record data
		ImGui::Checkbox(KEY("Preview", folder), &preview);
		ImGui::SameLine();
		ImGui::Checkbox(KEY("autoplay", folder), &autoPlayPreview);
		for (auto device = cameras.begin(); device != cameras.end(); device++) {
			device->camera->enableUpdateFrame = !preview;
		}
		if (preview && cameras.size() > 0 && getSize() > 0) {
			ImGui::SliderInt(KEY("previewIndex", folder), &nextpreviewIndex, 0, getSize() - 1);
			if (nextpreviewIndex != currentPreviewIndex) {
				for (auto device = cameras.begin(); device != cameras.end(); device++) {
					memcpy(
						(uint16_t*)(void*)device->camera->p_depth_frame,
						getBuffer(device->camera->serial)->at(nextpreviewIndex).p_depth_frame,
						device->camera->width * device->camera->height * sizeof(uint16_t)
					);

					memcpy(
						(unsigned char*)(void*)device->camera->p_color_frame,
						getBuffer(device->camera->serial)->at(nextpreviewIndex).colorRaw,
						INPUT_COLOR_CHANNEL * device->camera->width * device->camera->height * sizeof(unsigned char)
					);
					device->camera->frameNeedsUpdate = true;
				}
				currentPreviewIndex = nextpreviewIndex;
			}
			if (autoPlayPreview) {
				nextpreviewIndex++;
				if (nextpreviewIndex >= getSize()) {
					nextpreviewIndex = 0;
				}
			}
		}

		ImGui::Text("Trim Recorded Data :");
		if (ImGui::Button(KEY("Set currenPreview as Begin", folder))) {
			for (int i = 0; i < currentPreviewIndex; i++) {
				for (auto device = cameras.begin(); device != cameras.end(); device++) {
					auto recordFrames = getBuffer(device->camera->serial);
					free(recordFrames->at(0).p_depth_frame);
					free(recordFrames->at(0).colorRaw);
					free(recordFrames->at(0).xy_table);
					recordFrames->erase(recordFrames->begin());
				}
			}
		}
		ImGui::SameLine();
		if (ImGui::Button(KEY("Set currenPreview as Last", folder))) {
			for (auto device = cameras.begin(); device != cameras.end(); device++) {
				auto recordFrames = getBuffer(device->camera->serial);
				while (recordFrames->size() > currentPreviewIndex) {
					free(recordFrames->at(recordFrames->size() - 1).p_depth_frame);
					free(recordFrames->at(recordFrames->size() - 1).colorRaw);
					free(recordFrames->at(recordFrames->size() - 1).xy_table);
					recordFrames->pop_back();
				}
			}
		}
	}
	else {
		ImGui::Text(("Progress :" + std::to_string(alreadySavedFileCount) + "/" + std::to_string(buffers.size()*getSize())).c_str());
	}
}

bool CameraRecorder::clearBuffer() {
	if (state == SAVING)return false;
	for (auto buffer : buffers) {
		auto frames = *buffer.second;
		for (auto data : frames) {
			free(data.colorRaw);
			free(data.xy_table);
			free(data.p_depth_frame);
		}
		frames.clear();

	}
	for (auto buffer : buffers) {
		free(buffer.second);
	}
	buffers.clear();
	return true;
}
void CameraRecorder::saveFiles() {
	state = SAVING;
	cv::utils::fs::createDirectory(folder);

	std::vector<std::string> camserials;
	for (auto buffer : buffers) {
		auto frames = *buffer.second;

		std::vector<std::string> camfilenames;

		for (auto data : frames) {
			auto camera = data.camera;

			// save every camera's data
			std::ostringstream stringStream;
			stringStream << "./" << camera->serial << "_t" << std::setw(8) << std::setfill('0') << camfilenames.size();
			std::string filename = cv::utils::fs::join(folder, stringStream.str());
			camfilenames.push_back(stringStream.str() + ".json");

			JsonUtils::saveRealsenseJson(
				filename,
				camera->width, camera->height,
				camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
				camera->intri.depth_scale, data.p_depth_frame, data.colorRaw, data.xy_table,
				camera->farPlane, {
					camera->esitmatePlaneCenter.x,
					camera->esitmatePlaneCenter.y,
					camera->esitmatePlaneCenter.z,
					camera->esitmatePlaneNormal.x,
					camera->esitmatePlaneNormal.y,
					camera->esitmatePlaneNormal.z,
					camera->point2floorDistance
				}
			);
			free(data.colorRaw);
			free(data.xy_table);
			free(data.p_depth_frame);
			alreadySavedFileCount++;
		}

		camserials.push_back("./" + buffer.first + ".json");
		JsonUtils::saveSingleCamTimeInfo(
			cv::utils::fs::join(folder, buffer.first),
			camfilenames
		);

		frames.clear();
	}

	JsonUtils::saveMultiCamSyncInfo(
		cv::utils::fs::join(folder, folder),
		camserials
	);

	for (auto buffer : buffers) {
		free(buffer.second);
	}
	buffers.clear();
	state = DONE;
}
void CameraRecorder::exportBuffer2files() {
	if (state == INMEMORY) {
		exportBuffer2filesThread = std::thread(&CameraRecorder::saveFiles, this);
	}
}

int CameraRecorder::getSize() {
	if (buffers.size() > 0) {
		return buffers.begin()->second->size();
	}
	return 0;
}
std::vector<MannulBufferingFrame>* CameraRecorder::getBuffer(std::string serialnum) {
	if (buffers.find(serialnum) == buffers.end()) {
		buffers[serialnum] = new std::vector<MannulBufferingFrame>();
	}
	return buffers[serialnum];
}

CameraRecorder::CameraRecorder() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream sstr;
	sstr << "./" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
	folder = sstr.str();
}
CameraRecorder::~CameraRecorder() {
	for (auto buffer : buffers) {
		free(buffer.second);
	}

	if (state == DONE) {
		exportBuffer2filesThread.join();
	}
}

