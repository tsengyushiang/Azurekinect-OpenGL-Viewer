#include "./VirtualRouteAnimator.h"

VirtualRouteAnimator::VirtualRouteAnimator() 
	:fps(30), durationSecond(3) 
{
	// init
	running = false;
	currentFrame = 0;
}

int VirtualRouteAnimator::animeVirtualCamPose(SphericalCamPose& virtualCamPose){

	if (!running)return -1;

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
	return frame;
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

	if (ImGui::Button("Run Animation")) {
		running = true;
	}
}
