#include "./jsonUtils.h"
#include "../InputCamera/InputBase.h"
#include <opencv2/core/utils/filesystem.hpp>

template<typename T>
void Jsonformat::to_json(json& j, const T& p) {
	j = json{ {"id", p.id}, {"extrinsic", p.extrinsic} };
}

template<typename T>
void Jsonformat::from_json(const json& j, T& p) {
	j.at("id").get_to(p.id);
	j.at("extrinsic").get_to(p.extrinsic);
}

std::string JsonUtils::cameraPoseFilename = "CameraExtrinsics";
void JsonUtils::loadCameraPoses(
	std::string filename,
	std::vector<Jsonformat::CamPose>& poses
) {
	// write prettified JSON to another file
	std::ifstream i(filename);
	json j;
	i >> j;
	try {
		j = j["camExtrinsics"];
	}
	catch (...) {

	}
	for (json cam : j) {
		poses.push_back(cam);
	}

	i.close();
}

void JsonUtils::loadVirtualCam(
	std::string filename,
	std::vector<Jsonformat::CamPose>& poses,
	int& width, int& height,
	float& fx, float& fy, float& ppx, float& ppy
) {
	// write prettified JSON to another file
	std::ifstream i(filename);
	json j;
	i >> j;
	std::string frameFileName = j["realCamsRef"][0];

	json extrinsics = j["camExtrinsics"];

	for (json cam : extrinsics) {
		poses.push_back(cam);
	}
	i.close();

	std::string directory;

	directory = filename.substr(0, filename.rfind('\\')) + "\\" + frameFileName.substr(2, frameFileName.size());

	std::ifstream framefile(directory);
	json frameJson;
	framefile >> frameJson;
	width = frameJson["width"];
	height = frameJson["height"];
	fx = frameJson["fx"];
	fy = frameJson["fy"];
	ppx = frameJson["ppx"];
	ppy = frameJson["ppy"];
	framefile.close();

}

void JsonUtils::saveCameraPoses(
	std::vector<Jsonformat::CamPose>& poses
) {
	json j;
	for (Jsonformat::CamPose p : poses) {
		j.push_back(p);
	}

	// write prettified JSON to another file
	std::ofstream o(cameraPoseFilename + ".json");
	o << std::setw(4) << j << std::endl;
	o.close();
}


void JsonUtils::loadResolution(
	std::string filename,
	int& width, int& height
) {
	// assume is single frame file
	std::ifstream i(filename);
	json j;
	i >> j;	
	
	if (j.contains("width") && j.contains("height")) {
		width = j["width"];
		height = j["height"];
		i.close();
		return;
	}		
	else {
		i.close();

		// not single frame file
		std::string datafilename = j["realCamsRef"][0];

		std::string folderName = filename.substr(0, filename.find_last_of("\\/"));
		std::ifstream frame(cv::utils::fs::join(folderName, datafilename));
		json data;
		frame >> data;

		width = data["width"];
		height = data["height"];
		frame.close();
	}

}


bool JsonUtils::loadRealsenseJson(
	std::string filename,
	int& width, int& height,
	float& fx, float& fy, float& ppx, float& ppy, int& frameLength,
	float& depthscale, uint16_t** depthmap, unsigned char** colormap,float** xytable,
	float& farplane,float& plane_cx, float& plane_cy, float& plane_cz, float& plane_nx, float& plane_ny, float& plane_nz, float& plane_threshold
) {
	std::ifstream i(filename);
	json j;
	i >> j;

	depthscale = j["depthscale"];
	width = j["width"];
	height = j["height"];
	fx = j["fx"];
	fy = j["fy"];
	ppx = j["ppx"];
	ppy = j["ppy"];

	if (j.contains("far-plane")) {
		farplane = j["far-plane"];
	}

	if (j.contains("floor-plane")) {
		json plane = j["floor-plane"];
		plane_cx = plane["cx"];
		plane_cy = plane["cy"];
		plane_cz = plane["cz"];
		plane_nx = plane["nx"];
		plane_ny = plane["ny"];
		plane_nz = plane["nz"];
		plane_threshold = plane["threshold"];
	}

	std::cout << std::endl << filename<<" has : "<< depthscale<<" "<< width << " " << height <<"..."<< std::endl;

	bool hasXYtable = j.contains("xy_table");
	
	const int INPUT_JSON_COLOR_CHANNEL = 3;
	for (int i = 0; i < width * height; i++) {
		
		//load xy table
		if (hasXYtable) {
			(*xytable)[i * 2 + 0] = j["xy_table"][i * 2 + 0];
			(*xytable)[i * 2 + 1] = j["xy_table"][i * 2 + 1];
		}

		// load depth
		(*depthmap)[i] = j["depthmap_raw"][i];

		// load color
		for (int channel = 0; channel < INPUT_JSON_COLOR_CHANNEL; channel++) {
			(*colormap)[i* INPUT_COLOR_CHANNEL + channel] = 
				j["colormap_raw"][i * INPUT_JSON_COLOR_CHANNEL + channel];
		}
	}

	i.close();
	return hasXYtable;
}

void JsonUtils::saveMultiCamSyncInfo(
	std::string filename,
	std::vector<std::string> camTimefilenames
) {
	json j = {
		{"realCamsRef",camTimefilenames},
		{"realCamNum",camTimefilenames.size()}
	};

	// write prettified JSON to another file
	std::ofstream o(filename + ".json");
	o << std::setw(4) << j << std::endl;
	o.close();
}

bool JsonUtils::loadSinglCamTimeSequence(
	std::string filename,
	std::vector<std::string>& datafilenames
) {
	std::ifstream i(filename);
	json j;
	i >> j;

	if (!j.contains("realCamsRef")) {
		i.close();
		return false;
	}
	else {
		datafilenames.clear();
		std::string folderName = filename.substr(0, filename.find_last_of("\\/"));
		for (json cam : j["realCamsRef"]) {
			std::cout << cv::utils::fs::join(folderName, cam) << std::endl;
			datafilenames.push_back(cv::utils::fs::join(folderName, cam));
		}
	}

	i.close();
	return true;
}

void JsonUtils::saveSingleCamTimeInfo(
	std::string filename,
	std::vector<std::string> datafilenames
) {
	json j = {
		{"realCamsRef",datafilenames},
		{"totalFrame",datafilenames.size()},
	};

	// write prettified JSON to another file
	std::ofstream o(filename + ".json");
	o << std::setw(4) << j << std::endl;
	o.close();
}

void JsonUtils::saveRealsenseJson(
	std::string filename,
	int width, int height,
	float fx, float fy, float ppx, float ppy,
	float depthscale, const unsigned short* depthmap, const unsigned char* colormap, float* xy_table,
	float visulizeFarplane, Plane plane
) {
	std::vector<float> depthmap_raw;
	std::vector<float> xytable;
	std::vector<unsigned char> colormap_raw;
	std::vector<unsigned char> mask_raw;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int cindex = i * width + j;
			int index = i * width + j;

			depthmap_raw.push_back(depthmap[index]);
			xytable.push_back(xy_table[index*2+0]);
			xytable.push_back(xy_table[index*2+1]);
			colormap_raw.push_back(colormap[cindex * INPUT_COLOR_CHANNEL + 0]);
			colormap_raw.push_back(colormap[cindex * INPUT_COLOR_CHANNEL + 1]);
			colormap_raw.push_back(colormap[cindex * INPUT_COLOR_CHANNEL + 2]);
			//mask_raw.push_back(colormap[cindex * INPUT_COLOR_CHANNEL + 3]);
		}
	}

	json p = {
		{"cx",plane.cx},
		{"cy",plane.cy},
		{"cz",plane.cz},
		{"nx",plane.nx},
		{"ny",plane.ny},
		{"nz",plane.nz},
		{"threshold",plane.threshold},
	};

	json j = {
		{"width",width},
		{"height",height},
		{"fx",fx},
		{"fy",fy},
		{"ppx",ppx},
		{"ppy",ppy},
		{"depthscale",depthscale},
		{"colormap_raw",colormap_raw},
		{"depthmap_raw",depthmap_raw},
		{"xy_table",xytable},
		//{"yzCullingMask",mask_raw},
		{"frameLength",1},
		{"floor-plane",p},
		{"far-plane",visulizeFarplane}
	};

	//cv::Mat image(cv::Size(width, height), CV_8UC3, (void*)colormap_raw.data(), cv::Mat::AUTO_STEP);
	//cv::imwrite(filename + +".color.png", image);

	// write prettified JSON to another file
	std::ofstream o(filename +".json");
	o << j << std::endl;
	o.close();
}

void JsonUtils::saveRealsenseJson(
	std::string filename,
	int width, int height,
	float fx, float fy, float ppx, float ppy,
	float depthscale, float* depthmap, unsigned char* colormap,
	std::vector<float> extrinsic4x4
) {
	std::vector<float> depthmap_raw;
	std::vector<unsigned char> colormap_raw;

	for (int i = 0; i < width * height; i++) {
		depthmap_raw.push_back(depthmap[i] * depthscale);
		colormap_raw.push_back(colormap[i * INPUT_COLOR_CHANNEL + 0]);
		colormap_raw.push_back(colormap[i * INPUT_COLOR_CHANNEL + 1]);
		colormap_raw.push_back(colormap[i * INPUT_COLOR_CHANNEL + 2]);
	}

	json j = {
		{"width",width},
		{"height",height},
		{"fx",fx},
		{"fy",fy},
		{"ppx",ppx},
		{"ppy",ppy},
		{"depthscale",depthscale},
		{"colormap_raw",colormap_raw},
		{"depthmap_raw",depthmap_raw},
		{"extrinsic",extrinsic4x4},
		{"frameLength",1}
	};

	// write prettified JSON to another file
	std::ofstream o(filename + ".json");
	o << std::setw(4) << j << std::endl;
	o.close();
}