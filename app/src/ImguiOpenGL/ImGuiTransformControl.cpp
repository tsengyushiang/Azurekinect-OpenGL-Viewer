
#include "./ImGuiTransformControl.h"

TransformContorl::TransformContorl():mCurrentGizmoMode(ImGuizmo::LOCAL) {
}

bool TransformContorl::addWidget(glm::mat4& View, glm::mat4& Projection) {
	int lastUsing = 0;
	int matId = 0;

	ImGuizmo::SetID(matId);
	editTransform(&View[0][0], &Projection[0][0]);
	if (ImGuizmo::IsUsing())
	{
		lastUsing = matId;
	}
	return ImGuizmo::IsUsing();	
}

void TransformContorl::detachMatrix4() {
	matrix = nullptr;
}

void TransformContorl::attachMatrix4(glm::mat4& m4) {
	matrix = &m4[0][0];
}

void TransformContorl::addMenu() {
	if (matrix == nullptr)return;

	if (ImGuizmo::IsUsing())
	{
		ImGui::Text("Using gizmo");
	}
	else {
		ImGui::Text("Not Using gizmo");
	}

	{
		if (ImGui::Button("detach Matrix4")) {
			detachMatrix4();
			return;
		}
		if (ImGui::IsKeyPressed(90))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		if (ImGui::IsKeyPressed(69))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		if (ImGui::IsKeyPressed(82)) // r Key
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		/*if (ImGui::RadioButton("Universal", mCurrentGizmoOperation == ImGuizmo::UNIVERSAL))
			mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;*/
		float matrixTranslation[3], matrixRotation[3], matrixScale[3];
		ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
		ImGui::InputFloat3("Tr", matrixTranslation);
		ImGui::InputFloat3("Rt", matrixRotation);
		ImGui::InputFloat3("Sc", matrixScale);
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

		if (mCurrentGizmoOperation != ImGuizmo::SCALE)
		{
			if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
				mCurrentGizmoMode = ImGuizmo::LOCAL;
			ImGui::SameLine();
			if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
				mCurrentGizmoMode = ImGuizmo::WORLD;
		}
	}
}

void TransformContorl::editTransform(float* cameraView, float* cameraProjection)
{
	if (matrix==nullptr)return;
	ImGuiIO& io = ImGui::GetIO();

	ImGuizmo::SetDrawlist();
	float windowWidth = (float)ImGui::GetWindowWidth();
	float windowHeight = (float)ImGui::GetWindowHeight();
	ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

	ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix);

	ImGui::End();
}