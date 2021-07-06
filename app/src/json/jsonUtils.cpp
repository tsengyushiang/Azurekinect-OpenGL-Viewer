#include "./jsonUtils.h"

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
	std::ifstream i(filename + ".json");
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
	std::ifstream i(filename + ".json");
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
	std::ifstream i(filename + ".json");
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
	std::ifstream i(filename + ".json");
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

	*depthmap = (uint16_t*)calloc(width * height * frameLength, sizeof(uint16_t));
	*colormap = (unsigned char*)calloc(3 * width * height * frameLength, sizeof(unsigned char));

	for (int frame = 0; frame < frameLength; frame++) {

		std::cout << "frame" << frame << std::endl;
		for (int i = 0; i < width * height; i++) {
			int index = frame * width * height + i;
			(*depthmap)[index] = j["depthmap_raw"][index];
		}

		for (int i = 0; i < 3 * width * height; i++) {
			int index = frame * width * height * 3 + i;
			(*colormap)[index] = j["colormap_raw"][index];
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
		colormap_raw.push_back(colormap[i * 3 + 0]);
		colormap_raw.push_back(colormap[i * 3 + 1]);
		colormap_raw.push_back(colormap[i * 3 + 2]);
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
		{"depthmap_raw",depthmap_raw}
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
	float depthscale, float* depthmap, unsigned char* colormap,
	std::vector<float> extrinsic4x4
) {
	std::vector<float> depthmap_raw;
	std::vector<unsigned char> colormap_raw;

	for (int i = 0; i < width * height; i++) {
		depthmap_raw.push_back(depthmap[i] * depthscale);
		colormap_raw.push_back(colormap[i * 3 + 0]);
		colormap_raw.push_back(colormap[i * 3 + 1]);
		colormap_raw.push_back(colormap[i * 3 + 2]);
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
		{"extrinsic",extrinsic4x4}
	};

	// write prettified JSON to another file
	std::ofstream o(filename + ".json");
	o << std::setw(4) << j << std::endl;
	o.close();
}