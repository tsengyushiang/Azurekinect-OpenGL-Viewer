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

const std::string STATE_MSG[3] = {
	" (in memory)",
	" (saving)",
	" (done)"
};
enum STATE {
	INMEMORY=0,
	SAVING=1,
	DONE=2
};

class CameraRecorder {
	STATE state = INMEMORY;
	bool autoPlayPreview = true;
	bool preview = false;
	int currentPreviewIndex = -1;
	int nextpreviewIndex = -1;

	bool recordingMode = false;
	std::map<std::string, std::vector<MannulBufferingFrame>*>buffers;

	std::vector<MannulBufferingFrame>* getBuffer(std::string);
	int getSize();

	int alreadySavedFileCount = 0;
	std::thread exportBuffer2filesThread;
	void saveFiles();

public:
	std::string folder;
	bool clearBuffer();
	void exportBuffer2files();

	CameraRecorder();
	~CameraRecorder();
	void addGUI(std::vector<CameraGL> camera);
};