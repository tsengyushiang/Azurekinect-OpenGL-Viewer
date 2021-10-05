#include "./VirtualRouteAnimator.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <filesystem>
#include <iomanip>
#include <ctime>

VirtualRouteAnimator::VirtualRouteAnimator()
{
	// init
	running = false;
	currentFrame = 0;
}

RecordProgress VirtualRouteAnimator::animeVirtualCamPose(SphericalCamPose& virtualCamPose){

	if (!running)return {
		-1,
		-1
	};

	int frame = currentFrame;
	int totalFrame = fps * durationSecond;
	float currentPercentage = float(currentFrame) / totalFrame;

	virtualCamPose.AzimuthAngle = glm::mix(startPose.AzimuthAngle, endPose.AzimuthAngle, currentPercentage);
	virtualCamPose.PolarAngle = glm::mix(startPose.PolarAngle, endPose.PolarAngle, currentPercentage);
	virtualCamPose.distance = glm::mix(startPose.distance, endPose.distance, currentPercentage);
	virtualCamPose.lookAtPoint = glm::mix(startPose.lookAtPoint, endPose.lookAtPoint, currentPercentage);

	currentFrame++;
	if (currentFrame > totalFrame) {
		running = false;
		currentFrame = 0;
	}
	return {
		frame,
		currentPercentage
	};
}

void VirtualRouteAnimator::addUI(SphericalCamPose& virtualCam) {

	ImGui::Text("Virtual Route Animation :");

	if (ImGui::Button("Save Current pose as StartPose")) {
		startPose = virtualCam;
	}
	ImGui::Text("azAngle/pAngle/distance : %.1f/%.1f/%.1f", startPose.AzimuthAngle, startPose.PolarAngle, startPose.distance);

	if (ImGui::Button("Save Current pose as EndPose")) {
		endPose = virtualCam;
	}
	ImGui::Text("azAngle/pAngle/distance : %.1f/%.1f/%.1f", endPose.AzimuthAngle, endPose.PolarAngle, endPose.distance);

	ImGui::SliderInt("fps", &fps, 0, 144);
	ImGui::SliderInt("durationSecond", &durationSecond, 0, 10);

	if (ImGui::Button("Run Animation")) {
		running = true;

		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		std::ostringstream sstr;
		sstr << "./" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S")<<"-warppingResult-fps"<<fps<<"-duration"<<durationSecond;
		folder = sstr.str();
		cv::utils::fs::createDirectory(folder);
	}
}
