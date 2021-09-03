#pragma once
#include "./VirtualCam.h"
#include <iostream>
#include <cmath>

class VirtualRouteAnimator {
	SphericalCamPose startPose = {0,0,0};
	SphericalCamPose endPose = {0,0,0};
	int fps = 30, durationSecond = 3;
	bool running = false;
	int currentFrame = 0;
public:
	VirtualRouteAnimator();
	
	/*
	virtualCamPose : virtual camera pose to be set
	return current framenumber if not running reutrn -1
	*/
	int animeVirtualCamPose(SphericalCamPose& virtualCamPose);

	std::string folder;
	void addUI(SphericalCamPose& virtualCam);
};