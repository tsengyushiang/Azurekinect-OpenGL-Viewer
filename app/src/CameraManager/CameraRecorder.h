#pragma once

#include "./CameraGL.h"
#include "../InputCamera/Realsense.h"
#include "../InputCamera/JsonData.h"
#include "../InputCamera/AzureKinect.h"
#include <functional>

typedef struct MannulBufferingFrame {
	InputBase* camera;
	unsigned char* colorRaw;
	uint16_t* p_depth_frame;
	float* xy_table;
}MannulBufferingFrame;

class CameraRecorder {

	int bufferFramesCount = 0;
	std::map<std::string, std::vector<MannulBufferingFrame>*>buffers;

	std::vector<MannulBufferingFrame>* getBuffer(std::string);
	void exportBuffer2files();

public:
	~CameraRecorder();
	void addGUI(std::vector<CameraGL> camera);

};