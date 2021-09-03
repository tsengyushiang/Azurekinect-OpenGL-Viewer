#pragma once

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include "./Imgui/ImGuizmo.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);

class TransformContorl {

	float* matrix = nullptr;
	void editTransform(float* cameraView, float* cameraProjection);

public:

	ImGuizmo::MODE mCurrentGizmoMode;
	TransformContorl();
	void detachMatrix4();
	void attachMatrix4(glm::mat4& matrix);
	void addMenu();
	bool addWidget(glm::mat4& View, glm::mat4& Projection);
};