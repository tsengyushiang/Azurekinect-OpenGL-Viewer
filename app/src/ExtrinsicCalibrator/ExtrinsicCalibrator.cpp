#include "./ExtrinsicCalibrator.h"

void ExtrinsicCalibrator::addUI() {
	ImGui::Checkbox("calibrateFloorMode",&calibrateFloorMode);
	ImGui::SliderFloat("Distance Threshold", &collectthreshold, 0.05f, 0.3f);
	ImGui::SliderInt("ExpectCollect Point Count", &collectPointCout, 3, 50);
	if (calibrator != nullptr) {
		if (ImGui::Button("cancel calibrate")) {
			delete calibrator;
			calibrator = nullptr;
		}
	}
}

void ExtrinsicCalibrator::fitMarkersOnEstimatePlane(InputBase* camera) {

	std::vector<glm::vec2> corners = OpenCVUtils::getArucoMarkerCorners(
		camera->width,
		camera->height,
		camera->p_color_frame,
		INPUT_COLOR_CHANNEL
	);

	std::vector<Eigen::Vector3d> points;
	for (auto p : corners) {
		glm::vec3 point = camera->colorPixel2point(p);
		if (point.z != 0) {
			points.push_back(Eigen::Vector3d(point.x,point.y,point.z));
		}
	}

	if (points.size() > 3) {
		auto result = EigenUtils::best_plane_from_points(points);
		
		glm::vec3 centroid = glm::vec3(-result.centroid[0], -result.centroid[1], -result.centroid[2]);
		glm::vec3 planenormal = glm::vec3(result.plane_normal[0], result.plane_normal[1], result.plane_normal[2]);
		const glm::vec3 alignAxis = glm::vec3(0,1,0); // in my case (1, 0, 0)
		if ((planenormal.x * centroid.x + planenormal.y * centroid.y + planenormal.z * centroid.z) > 0) {
			planenormal = planenormal* -1.0f;
		}
		glm::vec3 v = glm::cross(alignAxis, planenormal);
		float angle = acos(glm::dot(alignAxis, planenormal) / (glm::length(alignAxis) * glm::length(planenormal)));
		glm::mat4 rotmat = glm::rotate(glm::mat4(1.0), angle, v);
		camera->esitmatePlaneCenter = -centroid;
		camera->esitmatePlaneNormal = planenormal;
		// calculate rotated plane normal to know error to xz-plane
		//glm::vec4 rotatePlaneNormal = rotmat * glm::vec4(result.plane_normal[0], result.plane_normal[1], result.plane_normal[2], 1);
		//std::cout << rotatePlaneNormal.x << " " << rotatePlaneNormal.y << " " << rotatePlaneNormal.z << std::endl;

		camera->modelMat = rotmat * glm::translate(glm::mat4(1.0), centroid);
	}
}


void ExtrinsicCalibrator::render(glm::mat4 mvp,GLuint shader_program) {
	if (calibrator != nullptr) {
		calibrator->render(mvp, shader_program);
	}
}

// detect aruco to calibrate unregisted camera
void ExtrinsicCalibrator::waitCalibrateCamera(
	std::vector<CameraGL>::iterator device,
	std::vector<CameraGL>& allDevice
) {
	if (calibrateFloorMode) {
		fitMarkersOnEstimatePlane(device->camera);
	}

	if (!device->camera->calibrated) {

		if (calibrator == nullptr) {
			alignDevice2calibratedDevice(device->camera, allDevice);
		}
		else {
			collectCalibratePoints();
		}
	}
}

void ExtrinsicCalibrator::collectCalibratePoints() {

	std::vector<glm::vec2> cornerSrc = OpenCVUtils::getArucoMarkerCorners(
		calibrator->sourcecam->width,
		calibrator->sourcecam->height,
		calibrator->sourcecam->p_color_frame,
		INPUT_COLOR_CHANNEL
	);

	std::vector<glm::vec2> cornerTrg = OpenCVUtils::getArucoMarkerCorners(
		calibrator->targetcam->width,
		calibrator->targetcam->height,
		calibrator->targetcam->p_color_frame,
		INPUT_COLOR_CHANNEL
	);

	if (cornerSrc.size() > 0 && cornerTrg.size() > 0) {
		if (calibrator->vaildCount == calibrator->size) {
			calibrator->calibrate();
			delete calibrator;
			calibrator = nullptr;
		}
		else {
			glm::vec3 centerSrc = glm::vec3(0, 0, 0);
			int count = 0;
			for (auto p : cornerSrc) {
				glm::vec3 point = calibrator->sourcecam->colorPixel2point(p);
				if (point.z == 0)	return;
				centerSrc += point;
				count++;
			}
			centerSrc /= count;

			glm::vec3 centerTrg = glm::vec3(0, 0, 0);
			count = 0;
			for (auto p : cornerTrg) {
				glm::vec3 point = calibrator->targetcam->colorPixel2point(p);
				if (point.z == 0)	return;
				centerTrg += point;
				count++;
			}
			centerTrg /= count;

			glm::vec4 src = calibrator->sourcecam->modelMat * glm::vec4(centerSrc.x, centerSrc.y, centerSrc.z, 1.0);
			glm::vec4 dst = calibrator->targetcam->modelMat * glm::vec4(centerTrg.x, centerTrg.y, centerTrg.z, 1.0);

			bool result = calibrator->pushCorrepondPoint(
				glm::vec3(src.x, src.y, src.z),
				glm::vec3(dst.x, dst.y, dst.z)
			);
		}
	}
}

void ExtrinsicCalibrator::alignDevice2calibratedDevice(InputBase* uncalibratedCam, std::vector<CameraGL>& allDevice) {

	InputBase* baseCamera = nullptr;
	glm::mat4 baseCam2Markerorigion;
	for (auto device = allDevice.begin(); device != allDevice.end(); device++) {
		if (device->camera->calibrated) {
			CalibrateResult c = putAruco2Origion(device->camera);
			if (c.success) {
				baseCamera = device->camera;
				baseCam2Markerorigion = c.calibMat;
				break;
			}
		}
	}
	if (baseCamera) {
		CalibrateResult c = putAruco2Origion(uncalibratedCam);
		if (c.success) {
			uncalibratedCam->modelMat = baseCamera->modelMat * glm::inverse(baseCam2Markerorigion) * c.calibMat;
			calibrator = new CorrespondPointCollector(uncalibratedCam, baseCamera, collectPointCout, collectthreshold);
		}
	}
}

CalibrateResult ExtrinsicCalibrator::putAruco2Origion(InputBase* camera) {

	CalibrateResult result;
	result.success = false;

	// detect aruco and put tag in origion
	std::vector<glm::vec2> corner = OpenCVUtils::getArucoMarkerCorners(
		camera->width,
		camera->height,
		camera->p_color_frame,
		INPUT_COLOR_CHANNEL
	);

	if (corner.size() > 0) {
		for (auto p : corner) {
			glm::vec3 point = camera->colorPixel2point(p);
			if (point.z == 0) {
				return result;
			}
			result.points.push_back(point);
		}
		glm::vec3 x = glm::normalize(result.points[0] - result.points[1]);
		glm::vec3 z = glm::normalize(result.points[2] - result.points[1]);
		glm::vec3 y = glm::vec3(
			x.y * z.z - x.z * z.y,
			x.z * z.x - x.x * z.z,
			x.x * z.y - x.y * z.x
		);
		glm::mat4 tranform = glm::mat4(
			x.x, x.y, x.z, 0.0,
			-y.x, -y.y, -y.z, 0.0,
			z.x, z.y, z.z, 0.0,
			result.points[1].x, result.points[1].y, result.points[1].z, 1.0
		);
		result.success = true;
		result.calibMat = glm::inverse(tranform);

		// draw xyz-axis
		//GLfloat axisData[] = {
		//	//  X     Y     Z           R     G     B
		//		points[0].x, points[0].y, points[0].z,       0.0f, 0.0f, 0.0f,
		//		points[1].x, points[1].y, points[1].z,       1.0f, 0.0f, 0.0f,
		//		points[2].x, points[2].y, points[2].z,       0.0f, 1.0f, 0.0f,
		//		points[3].x, points[3].y, points[3].z,       0.0f, 0.0f, 1.0f,
		//};
		//ImguiOpeGL3App::setPointsVAO(axisVao, axisVbo, axisData, 4);
		//glm::mat4 mvp = Projection * View;
		//ImguiOpeGL3App::render(mvp, 10.0, shader_program, axisVao, 4, GL_POINTS);
	}
	return result;
}

bool ExtrinsicCalibrator::checkIsCalibrating(std::string serial, glm::vec3& index) {
	if (calibrator != nullptr) {
		if (serial == calibrator->sourcecam->serial) {
			index = glm::vec3(1, 0, 0);
			return true;
		}
		if (serial == calibrator->targetcam->serial) {
			index = glm::vec3(0, 1, 0);
			return true;
		}
	}
	return false;
}
