#include "./jsonUtils.h"
#include "../InputCamera/InputBase.h"

int SaveFileCount = 0;

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
	j = j["camExtrinsics"];

	for (json cam : j) {
		poses.push_back(cam);
	}

	i.close();
}

void JsonUtils::loadVirtualCam(
	std::string filename,
	std::vector<Jsonformat::CamPose>& poses,
	int& width, int& height, float& farPlane,
	float& fx, float& fy, float& ppx, float& ppy
) {
	// write prettified JSON to another file
	std::ifstream i(filename);
	json j;
	i >> j;
	width = j["width"];
	height = j["height"];
	farPlane = j["farPlane"];
	fx = j["fx"];
	fy = j["fy"];
	ppx = j["ppx"];
	ppy = j["ppy"];

	json extrinsics = j["camExtrinsics"];

	for (json cam : extrinsics) {
		poses.push_back(cam);
	}

	i.close();
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


void JsonUtils::loadRealsenseJson(
	std::string filename,
	int& width, int& height
) {
	std::ifstream i(filename);
	json j;
	i >> j;

	width = j["width"];
	height = j["height"];

	i.close();
}


void JsonUtils::loadRealsenseJson(
	std::string filename,
	int& width, int& height,
	float& fx, float& fy, float& ppx, float& ppy, int& frameLength,
	float& depthscale, uint16_t** depthmap, unsigned char** colormap
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
	frameLength = j["frameLength"];

	free(*depthmap);
	free(*colormap);
	*depthmap = (uint16_t*)calloc(width * height * frameLength, sizeof(uint16_t));
	*colormap = (unsigned char*)calloc(INPUT_COLOR_CHANNEL * width * height * frameLength, sizeof(unsigned char));

	const int INPUT_JSON_COLOR_CHANNEL = 3;
	for (int frame = 0; frame < frameLength; frame++) {

		std::cout << "frame" << frame << std::endl;
		for (int i = 0; i < width * height; i++) {
			int index = frame * width * height + i;
			(*depthmap)[index] = j["depthmap_raw"][index];
		}

		int frameBegin = frame * width * height * INPUT_JSON_COLOR_CHANNEL;
		for (int i = 0; i < width * height; i++) {
			for (int channel = 0; channel < INPUT_JSON_COLOR_CHANNEL; channel++) {
				(*colormap)[frameBegin + i* INPUT_COLOR_CHANNEL + channel] = 
					j["colormap_raw"][frameBegin + i * INPUT_JSON_COLOR_CHANNEL + channel];
			}
		}
	}

	i.close();
}

void JsonUtils::saveRealsenseJson(
	std::string filename,
	int width, int height,
	float fx, float fy, float ppx, float ppy,
	float depthscale, const unsigned short* depthmap, const unsigned char* colormap
) {
	std::vector<float> depthmap_raw;
	std::vector<unsigned char> colormap_raw;

	for (int i = 0; i < width * height; i++) {
		depthmap_raw.push_back(depthmap[i]);
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
		{"frameLength",1},
	};

	cv::Mat image(cv::Size(width, height), INPUT_COLOR_CHANNEL==3? CV_8UC3 : CV_8UC4, (void*)colormap, cv::Mat::AUTO_STEP);
	cv::imwrite(filename + std::to_string(SaveFileCount)+".png", image);

	// write prettified JSON to another file
	std::ofstream o(filename + std::to_string(SaveFileCount++) +".json");
	o << std::setw(4) << j << std::endl;
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