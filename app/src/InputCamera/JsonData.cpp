#include "./JsonData.h"

JsonData::JsonData(int w, int h) :InputBase(w, h, w, h) {
	framefiles.clear();
    autoUpdate = std::thread(&JsonData::getLatestFrame, this);
};

JsonData::~JsonData() {
	autoUpdate.join();
}

void JsonData::setFrameIndex(int index){
	syncTime = index;
	updateFrame();
}

void JsonData::updateFrame() {
	if (currentTime != syncTime && framefiles.size() > syncTime) {
		currentTime = syncTime;

		std::cout << framefiles.size() << " " << syncTime << std::endl;

		float planecx, planecy, planecz;
		float planenx, planeny, planenz;
		float planeThreshold = 0;
		bool hasXYtable = JsonUtils::loadRealsenseJson(
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
			&p_color_frame,
			&xy_table,
			farPlane,
			planecx, planecy, planecz,
			planenx, planeny, planenz,
			planeThreshold
		);

		if (hasXYtable) {
			cudaMemcpy(xy_table_cuda, xy_table, width * height * 2 * sizeof(float), cudaMemcpyHostToDevice);
			xy_tableReady = true;
		}
		else {
			setXYtable(intri.ppx, intri.ppy, intri.fx, intri.fy);
		}

		if (!floorEquationGot) {
			esitmatePlaneCenter = glm::vec3(planecx, planecy, planecz);
			esitmatePlaneNormal = glm::vec3(planenx, planeny, planenz);
			point2floorDistance = planeThreshold;
		}

		frameNeedsUpdate = true;
	}
}


void JsonData::getLatestFrame() {

	while (true) {
		try {
			updateFrame();
		}
		catch (...) {
			break;
		}
	}
}