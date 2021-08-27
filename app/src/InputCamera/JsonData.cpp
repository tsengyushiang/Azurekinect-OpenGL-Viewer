#include "./JsonData.h"

JsonData::JsonData(int w, int h) :InputBase(w, h, w, h) {
	framefiles.clear();
    autoUpdate = std::thread(&JsonData::getLatestFrame, this);
};

JsonData::~JsonData() {
	autoUpdate.join();
}

void JsonData::getLatestFrame() {

	while (true) {
		try {
			if (currentTime != syncTime && framefiles.size() > syncTime) {
				currentTime = syncTime;

				std::cout << framefiles.size()<<" "<< syncTime << std::endl;
				JsonUtils::loadRealsenseJson(
					framefiles[currentTime],
					width,
					height,
					intri.fx,
					intri.fy,
					intri.ppx,
					intri.ppy,
					frameLength,
					intri.depth_scale,
					&p_depth_frame,
					&p_color_frame);

				frameNeedsUpdate = true;
			}
		}
		catch (...) {
			break;
		}
	}
}