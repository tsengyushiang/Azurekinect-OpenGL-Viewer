#pragma once

#include <iomanip>
#include <iostream>
#include <fstream> 

#include "json/json.hpp"
using json = nlohmann::json;

class JsonUtils {

public:
	static void saveRealsenseJson(
		std::string filename, 
		int width,int height,
		float fx,float fy,float ppx,float py,
		float depthscale, const unsigned short* depthmap,const unsigned char* colormap,
		std::vector<float> extrinsic4x4
	);
};