#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include "../ImguiOpenGL/ImguiOpeGL3App.h"

class MetaDataExporter {

	bool isexporting = false;
public:
	int frameIndex;
	std::string folder;

	void addGui();

	void writeObjWithMaterial(
		float** posuvnormal,
		unsigned int** faceIndices,
		int width, int height, int facecount,
		cv::Mat color,
		std::string folder,
		std::string serial
	);

	void startExport();
	void stopExport();
	bool isExporting();
};