#pragma once

#include <iomanip>
#include <iostream>
#include <fstream> 
#include <opencv2/opencv.hpp>
#include "json/json.hpp"
using json = nlohmann::json;
namespace Jsonformat {
	struct CamPose {
		std::string id;
		std::vector<float> extrinsic;
	};

	template<typename T>
	void to_json(json& j, const T& p);
	template<typename T>
	void from_json(const json& j, T& p);
}

typedef struct Plane {
	float cx;
	float cy;
	float cz;
	float nx;
	float ny;
	float nz;
	float threshold;
}Plane;

class JsonUtils {

public:

	static void saveMultiCamSyncInfo(
		std::string filename,
		std::vector<std::string> camTimefilenames
	);

	static void saveSingleCamTimeInfo(
		std::string filename,
		std::vector<std::string> datafilenames
	);

	static bool loadSinglCamTimeSequence(
		std::string filename,
		std::vector<std::string>& datafilenames
	);

	static void saveRealsenseJson(
		std::string filename, 
		int width,int height,
		float fx,float fy,float ppx,float ppy,
		float depthscale, const unsigned short* depthmap,const unsigned char* colormap,float* xy_table,
		float visulizeFarplane, Plane plane
	);
	static void saveRealsenseJson(
		std::string filename, 
		int width,int height,
		float fx,float fy,float ppx,float ppy,
		float depthscale, float* depthmap,unsigned char* colormap,
		std::vector<float> extrinsic4x4
	);

	// return if xy_table is found.
	static bool loadRealsenseJson(
		std::string filename,
		int& width, int& height,
		float& fx, float& fy, float& ppx, float& ppy,
		int& frameLength, float& depthscale, uint16_t** depthmap, unsigned char** colormap, float** xytable,
		float& farplane, float& plane_cx, float& plane_cy, float& plane_cz, float& plane_nx, float& plane_ny, float& plane_nz, float& plane_threshold
	);
	static void loadResolution(
		std::string filename,
		int& width, int& height
	);
	static std::string cameraPoseFilename;
	static void saveCameraPoses(
		std::vector<Jsonformat::CamPose>& poses
	);
	static void loadCameraPoses(
		std::string filename,
		std::vector<Jsonformat::CamPose>& poses
	);

	static void loadVirtualCam(
		std::string filename,
		std::vector<Jsonformat::CamPose>& poses,
		int& width, int& height,
		float& fx, float& fy, float& ppx, float& ppy
	);
};