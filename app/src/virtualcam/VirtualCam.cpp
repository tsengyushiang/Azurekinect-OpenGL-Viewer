#include "VirtualCam.h"

glm::mat4 VirtualCam::getModelMat(glm::vec3 lookAtPoint, int curFrame) {

	if (isFromFile) {
		if (curFrame < 0 || modelMats.size() == 0)
			return glm::mat4(1.0);
		return modelMats[curFrame % modelMats.size()];
	}
	//manual mode
	glm::mat4 rt = glm::lookAt(
		glm::vec3(
			distance * sin(PolarAngle) * cos(AzimuthAngle) + lookAtPoint.x,
			distance * cos(PolarAngle) + lookAtPoint.y,
			distance * sin(PolarAngle) * sin(AzimuthAngle) + lookAtPoint.z), // Camera is at (4,3,3), in World Space
		lookAtPoint, // and looks at the origin
		glm::vec3(0, -1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
	);
	return glm::scale(glm::inverse(rt), glm::vec3(1, -1, -1));
}

VirtualCam::VirtualCam(int width, int height) : viewport(width, height), debugview(width, height){
	w = width;
	h = height;
	colorRaw = new uchar[w * h * 3];
	depthRaw = new float[w * h];
	depthintRaw = new uint16_t[w * h];
}

void VirtualCam::addUI() {


	ImGui::Begin("VirtualCam : ");

	ImGui::Text("Route: ");

	static char jsonfilename[100] = "CameraExtrinsics";
	ImGui::Text("jsonfilename: ");

	ImGui::SameLine();
	if (ImGui::Button("load")) {
		std::vector<Jsonformat::CamPose> camRoute;
		JsonUtils::loadVirtualCam(jsonfilename, camRoute,
			w, h, farplane, fx, fy, ppx, ppy);
		for (auto cam : camRoute) {
			modelMats.push_back(glm::mat4(
				cam.extrinsic[0], cam.extrinsic[4], cam.extrinsic[8], cam.extrinsic[12],
				cam.extrinsic[1], cam.extrinsic[5], cam.extrinsic[9], cam.extrinsic[13],
				cam.extrinsic[2], cam.extrinsic[6], cam.extrinsic[10], cam.extrinsic[14],
				cam.extrinsic[3], cam.extrinsic[7], cam.extrinsic[11], cam.extrinsic[15]
			));
			std::cout << cam.extrinsic[0] << ", " << cam.extrinsic[4] << "," << cam.extrinsic[8] << "," << cam.extrinsic[12] << std::endl;
			std::cout << cam.extrinsic[1] << ", " << cam.extrinsic[5] << "," << cam.extrinsic[9] << "," << cam.extrinsic[13] << std::endl;
			std::cout << cam.extrinsic[2] << ", " << cam.extrinsic[6] << "," << cam.extrinsic[10] << "," << cam.extrinsic[14] << std::endl;
			std::cout << cam.extrinsic[3] << ", " << cam.extrinsic[7] << "," << cam.extrinsic[11] << "," << cam.extrinsic[15] << std::endl;

		}

	}
	ImGui::SameLine();
	ImGui::InputText("##jsonfilenameurlInput", jsonfilename, 20);

	if(ImGui::Button("switch debug mode")) {
		std::cout << "manual mode" << isFromFile  << std::endl;
		isFromFile = !isFromFile;
	}

	if (!isFromFile) {
		ImGui::Text("Sphereical Coordiante :");
		ImGui::SliderFloat("farplane##virtualcam", &farplane, 0, 10);
		ImGui::SliderFloat("AzimuthAngle##virtualcam", &AzimuthAngle, AzimuthAnglemin, AzimuthAngleMax);
		ImGui::SliderFloat("PolarAngle##virtualcam", &PolarAngle, PolarAnglemin, PolarAngleMax);
		ImGui::SliderFloat("distance##virtualcam", &distance, distancemin, distanceMax);
	}

	ImGui::End();

}

void VirtualCam::save(std::string filename, glm::vec3 lookAtPoint, int curFrame) {
	glm::mat4 modelMat = getModelMat(lookAtPoint, curFrame);

	//for (int i = 0; i < w * h; i++) {
	//	if (dpixels[i] > 0.9999) {
	//		dpixels[i] = 0;
	//	}
	//}

	std::vector<float> extrinsic = {
		modelMat[0][0],modelMat[1][0],modelMat[2][0],modelMat[3][0],
		modelMat[0][1],modelMat[1][1],modelMat[2][1],modelMat[3][1],
		modelMat[0][2],modelMat[1][2],modelMat[2][2],modelMat[3][2],
		modelMat[0][3],modelMat[1][3],modelMat[2][3],modelMat[3][3]
	};

	JsonUtils::saveRealsenseJson(filename,
		w, h,
		fx, fy, ppx, ppy, farplane,
		depthRaw, colorRaw, extrinsic
	);
}

void VirtualCam::renderFrustum(
	glm::mat4 devicemvp,
	GLuint& vao, GLuint& vbo, GLuint render_vertexColor_program) 
{
	// render camera frustum
	ImguiOpeGL3App::genCameraHelper(
		vao, vbo,
		w, h,
		ppx, ppy, fx, fy,
		color, farplane, false
	);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	ImguiOpeGL3App::render(devicemvp, 0, render_vertexColor_program, vao, 3 * 4, GL_TRIANGLES);

}

