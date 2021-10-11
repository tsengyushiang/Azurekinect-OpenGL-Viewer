#include "./MetaDataExporter.h"
#include <opencv2/core/utils/filesystem.hpp>
#include "../config.h"

void MetaDataExporter::addGui() {
	if (ImGui::Button("Save Mesh")) {
		startExport();
	}
}

void MetaDataExporter::startExport() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream sstr;
	sstr << "./" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S") << "-PlaneMeshes";
	folder = sstr.str();
	cv::utils::fs::createDirectory(folder);
	
	frameIndex = 0;
	isexporting = true;
}
bool MetaDataExporter::isExporting() {

	return isexporting;
}
void MetaDataExporter::stopExport() {
	isexporting = false;
}

void MetaDataExporter::writeObjWithMaterial(
	float** posuvnormal,
	unsigned int** faceIndices,
	int width, int height, int facecount,
	cv::Mat color,
	std::string folder,
	std::string serial
) {
	std::string meshObj = serial+".obj";
	std::string meshMtl = serial +".mtl";
	std::string colorMapf = serial + ".png";

	std::ofstream file;
	file.open(cv::utils::fs::join(folder, meshObj));
	if (file.is_open()) {
		file << "mtllib " << meshMtl << std::endl;
		file << "usemtl " << meshMtl << std::endl;
		for (int index = 0; index < width * height; index++) {
			file << "v" << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 0] << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 1] << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_VERTEX + 2]
				<< std::endl;
			file << "vt" << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_UV + 0] << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_UV + 1]
				<< std::endl;
			file << "vn" << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 0] << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 1] << " "
				<< (*posuvnormal)[index * ATTRIBUTESIZE + ATTRIBUTE_OFFSET_NORMAL + 2]
				<< std::endl;
		}
		
		// add one for face indice in obj format
		for (int i = 0; i < facecount; i++) {
			file << "f" << " "
				<< (*faceIndices)[(i) * 3 + 0] + 1 << "/" << (*faceIndices)[(i) * 3 + 0] + 1 << "/" << (*faceIndices)[(i) * 3 + 0] + 1 << " "
				<< (*faceIndices)[(i) * 3 + 1] + 1 << "/" << (*faceIndices)[(i) * 3 + 1] + 1 << "/" << (*faceIndices)[(i) * 3 + 1] + 1 << " "
				<< (*faceIndices)[(i) * 3 + 2] + 1 << "/" << (*faceIndices)[(i) * 3 + 2] + 1 << "/" << (*faceIndices)[(i) * 3 + 2] + 1
				<< std::endl;
		}
		file.close();
	}
	else {
		std::cout << "Failed to open file : " << meshObj << std::endl;
	}	

	file.open(cv::utils::fs::join(folder, meshMtl));
	if (file.is_open()) {
		file << "newmtl " << meshMtl << std::endl;
		file << "map_Kd " << colorMapf << std::endl;
		file << "Ka 1.0 1.0 1.0"<< std::endl;
		file << "Kd 1.0 1.0 1.0"<< std::endl;
		file << "Ks 1.0 1.0 1.0"<< std::endl;
		file.close();
	}
	else {
		std::cout << "Failed to open file : " << meshMtl << std::endl;
	}

	cv::imwrite(cv::utils::fs::join(folder, colorMapf), color);
}