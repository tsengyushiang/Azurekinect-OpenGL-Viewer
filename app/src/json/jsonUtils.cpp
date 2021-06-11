#include "./jsonUtils.h"

template<typename T>
void Jsonformat::to_json(json& j, const T& p) {
	j = json{ {"id", p.id}, {"extrinsic", p.extrinsic}};
}

template<typename T>
void Jsonformat::from_json(const json& j, T& p) {
	j.at("id").get_to(p.id);
	j.at("extrinsic").get_to(p.extrinsic);
}

std::string JsonUtils::cameraPoseFilename = "CameraExtrinsics";
void JsonUtils::loadCameraPoses(
	std::vector<Jsonformat::CamPose>& poses
) {
	// write prettified JSON to another file
	std::ifstream i(cameraPoseFilename + ".json");
	json j;
	i >> j;

	for (json cam : j) {
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
	int& width, int& height,
	float& fx, float& fy, float& ppx, float& ppy,
	float& depthscale, uint16_t** depthmap, unsigned char** colormap
){
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
	
	*depthmap = (uint16_t*)calloc(width * height, sizeof(uint16_t));
	*colormap = (unsigned char*)calloc(3 * width * height, sizeof(unsigned char));

	for (int i = 0; i < width * height; i++) {
		(*depthmap)[i] = j["depthmap_raw"][i];
	}	
	
	for (int i = 0; i < 3*width * height; i++) {
		(*colormap)[i] = j["colormap_raw"][i];
	}

	i.close();
}

void JsonUtils::saveRealsenseJson(
	std::string filename,
	int width, int height,
	float fx, float fy, float ppx, float ppy,
	float depthscale, const unsigned short* depthmap,const unsigned char* colormap
){
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
	std::ofstream o(filename+".json");
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