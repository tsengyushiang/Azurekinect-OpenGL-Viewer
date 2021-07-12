#include "CameraGL.h"

CameraGL::CameraGL(InputBase* cam) :planemesh(cam->width, cam->height, INPUT_COLOR_CHANNEL), framebuffer(cam->width, cam->height) {
	camera = cam;
	CudaOpenGL::createCudaGLTexture(&image, &image_cuda, camera->width, camera->height);
	CudaOpenGL::createCudaGLTexture(&representColorImage, &representColorImage_cuda, camera->width, camera->height);
	CudaOpenGL::createCudaGLTexture(&depthvis, &depthvis_cuda, camera->width, camera->height);

}
void CameraGL::destory() {
	CudaOpenGL::deleteCudaGLTexture(&image, &image_cuda);
	CudaOpenGL::deleteCudaGLTexture(&representColorImage, &representColorImage_cuda);
	CudaOpenGL::deleteCudaGLTexture(&depthvis, &depthvis_cuda);

	planemesh.destory();
	free(camera);
}
void CameraGL::updateImages(
	ImVec4 chromaKeyColor,float chromaKeyColorThreshold,
	int maskErosionSize, bool autoDepthDilation,
	int curFrame
) 
{
	auto copyHost2Device = [this](const void* depthRaw, size_t depthSize, const void* colorRaw, size_t colorSize) {
		cudaMemcpy(planemesh.cudaDepthData, depthRaw, depthSize, cudaMemcpyHostToDevice);
		cudaMemcpy(planemesh.cudaColorData, colorRaw, colorSize, cudaMemcpyHostToDevice);
	};
	camera->fetchframes(copyHost2Device);

	// acutal texture
	CudaAlogrithm::chromaKeyBackgroundRemove(&image_cuda, planemesh.cudaColorData, camera->width, camera->height,
		glm::vec3(
			chromaKeyColor.x * 255,
			chromaKeyColor.y * 255,
			chromaKeyColor.z * 255
		), chromaKeyColorThreshold
	);

	CudaAlogrithm::maskErosion(&image_cuda, camera->width, camera->height, maskErosionSize);

	if (autoDepthDilation) {
		CudaAlogrithm::fillDepthWithDilation(&image_cuda, planemesh.cudaDepthData, camera->width, camera->height);
	}

	CudaAlogrithm::depthVisualize(&image_cuda, &depthvis_cuda, planemesh.cudaDepthData, camera->width, camera->height, camera->intri.depth_scale, camera->farPlane);

	//// debug : index map for project coverage
	CudaAlogrithm::chromaKeyBackgroundRemove(&representColorImage_cuda, planemesh.cudaColorData, camera->width, camera->height,
		glm::vec3(
			chromaKeyColor.x * 255,
			chromaKeyColor.y * 255,
			chromaKeyColor.z * 255
		), chromaKeyColorThreshold,
		glm::vec3(
			color.x * 255,
			color.y * 255,
			color.z * 255
		)
	);
}
// pass realsense data to cuda and compute plane mesh and point cloud
void CameraGL::updateMeshwithCUDA(float planeMeshThreshold, int depthDilationIterationCount) {
	CudaAlogrithm::depthMap2point(
		&planemesh.cuda_vbo_resource,
		planemesh.cudaDepthData, &image_cuda,
		camera->width, camera->height,
		camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
		camera->intri.depth_scale, camera->farPlane, depthDilationIterationCount);

	CudaAlogrithm::depthMapTriangulate(
		&planemesh.cuda_vbo_resource,
		&planemesh.cuda_ibo_resource,
		camera->width,
		camera->height,
		planemesh.cudaIndicesCount,
		planeMeshThreshold
	);
}

void CameraGL::save() {
	JsonUtils::saveRealsenseJson(
		camera->serial,
		camera->width, camera->height,
		camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
		camera->intri.depth_scale, camera->p_depth_frame, camera->p_color_frame
	);
}

void CameraGL::addui() {
	auto KEY = [this](std::string keyword)->const char* {
		return (keyword + std::string("##") + camera->serial).c_str();
	};
	if (ImGui::Button(KEY("stop"))) {
		ready2Delete = true;
	}
	ImGui::SameLine();
	ImGui::Text(camera->serial.c_str());
	ImGui::SameLine();
	ImGui::Checkbox(KEY("calibrated"), &(camera->calibrated));
	ImGui::ColorEdit3(KEY("color"), (float*)&color); // Edit 3 floats representing a color
	ImGui::SliderFloat(KEY("clip-z"), &camera->farPlane, 0.5f, 15.0f);
}

// render single realsense mesh
void CameraGL::renderMesh(glm::mat4& mvp, GLuint& program) {
	auto render = [this, mvp, program](GLuint& vao, int& count) {
		glm::mat4 m = mvp * camera->modelMat;
		ImguiOpeGL3App::renderElements(m, 0, program, vao, count * 3, GL_FILL);
	};

	planemesh.render(render);
}

void CameraGL::renderFrustum(
	glm::mat4 worldMVP, glm::vec3 camColor,
	GLuint& vao, GLuint& vbo,GLuint render_vertexColor_program,GLuint render_Texture_program
) {
	glm::mat4 deviceMVP = worldMVP * camera->modelMat;

	// render camera frustum
	ImguiOpeGL3App::genCameraHelper(
		vao, vbo,
		camera->width, camera->height,
		camera->intri.ppx, camera->intri.ppy, camera->intri.fx, camera->intri.fy,
		camColor, 0.2, false
	);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	ImguiOpeGL3App::render(deviceMVP, 0, render_vertexColor_program, vao, 3 * 4, GL_TRIANGLES);

	ImguiOpeGL3App::genCameraHelper(
		vao, vbo,
		camera->width, camera->height,
		camera->intri.ppx, camera->intri.ppy, camera->intri.fx, camera->intri.fy,
		camColor, 0.2, true
	);

	std::string uniformNames[] = { "color" };
	GLuint textureId[] = { image };
	ImguiOpeGL3App::activateTextures(render_Texture_program, uniformNames, textureId, 1);

	std::string outlinerRGB[] = { "outliner_r","outliner_g" ,"outliner_b" };
	float values[] = { color.x,color.y,color.z };
	ImguiOpeGL3App::setUniformFloats(render_Texture_program, outlinerRGB, values, 3);

	ImguiOpeGL3App::render(deviceMVP, 0, render_Texture_program, vao, 3 * 4, GL_TRIANGLES);
}