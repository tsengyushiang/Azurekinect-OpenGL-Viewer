#include "CameraGL.h"

GLFrameBuffer* CameraGL::getFrameBuffer(FrameBuffer type) {

	if (type == FrameBuffer::MASK) {
		return &maskInVirtualView;
	}
	else if (type == FrameBuffer::MESHNORMAL) {
		return &meshnormalInVirtualView;
	}
	else if (type == FrameBuffer::COSWEIGHT) {
		return &cosWeightInVirtualView;
	}
	else if (type == FrameBuffer::AFTERDISCARD) {
		return &afterDicardInVirtualView;
	}
}
glm::mat4 CameraGL::getModelMat() {
	return camera->modelMat;
}

CameraGL::CameraGL(InputBase* cam) :planemesh(cam->width, cam->height, INPUT_COLOR_CHANNEL), 
	maskInVirtualView(cam->width, cam->height),
	meshnormalInVirtualView(cam->width, cam->height),
	cosWeightInVirtualView(cam->width, cam->height),
	afterDicardInVirtualView(cam->width, cam->height)
{
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

void CameraGL::imagesPreprocessing(int maskErosionSize, bool autoDepthDilation) {

	CudaAlogrithm::maskErosion(&image_cuda, camera->width, camera->height, maskErosionSize);

	if (autoDepthDilation) {
		CudaAlogrithm::fillDepthWithDilation(&image_cuda, planemesh.cudaDepthData, camera->width, camera->height);
	}
}

void CameraGL::updateImages(
	ImVec4 chromaKeyColor,float chromaKeyColorThreshold
) 
{
	auto copyHost2Device = [this](const void* depthRaw, size_t depthSize, const void* colorRaw, size_t colorSize) {
		cudaMemcpy(planemesh.cudaDepthData, depthRaw, depthSize, cudaMemcpyHostToDevice);
		cudaMemcpy(planemesh.cudaColorData, colorRaw, colorSize, cudaMemcpyHostToDevice);
	};
	camera->fetchframes(copyHost2Device);

	// acutal texture and create Mask
	CudaAlogrithm::chromaKeyBackgroundRemove(&image_cuda, planemesh.cudaColorData, camera->width, camera->height,
		glm::vec3(
			chromaKeyColor.x * 255,
			chromaKeyColor.y * 255,
			chromaKeyColor.z * 255
		), chromaKeyColorThreshold
	);
	
	CudaAlogrithm::clipFloorAndFarDepth(
		&image_cuda, planemesh.cudaDepthData,
		camera->width, camera->height, camera->xy_table_cuda, camera->intri.depth_scale,
		camera->farPlane, camera->esitmatePlaneCenter, camera->esitmatePlaneNormal, camera->point2floorDistance
	);

	CudaAlogrithm::depthVisualize(&image_cuda, &depthvis_cuda, planemesh.cudaDepthData, camera->width, camera->height, camera->intri.depth_scale, camera->farPlane);	

	//// debug : index map for project coverage
	//CudaAlogrithm::chromaKeyBackgroundRemove(&representColorImage_cuda, planemesh.cudaColorData, camera->width, camera->height,
	//	glm::vec3(
	//		chromaKeyColor.x * 255,
	//		chromaKeyColor.y * 255,
	//		chromaKeyColor.z * 255
	//	), chromaKeyColorThreshold,
	//	glm::vec3(
	//		color.x * 255,
	//		color.y * 255,
	//		color.z * 255
	//	)
	//);
}
// pass realsense data to cuda and compute plane mesh and point cloud
void CameraGL::updateMeshwithCUDA(float planeMeshThreshold, int pointSmoothing) {
	CudaAlogrithm::depthMap2point(
		&planemesh.cuda_vbo_resource,
		planemesh.cudaDepthData, &image_cuda,
		camera->width, camera->height,camera->xy_table_cuda,
		camera->intri.depth_scale, camera->farPlane);

	CudaAlogrithm::planePointsLaplacianSmoothing(
		&planemesh.cuda_vbo_resource,
		camera->width, camera->height, pointSmoothing
	);

	CudaAlogrithm::planeVertexNormalEstimate(
		&planemesh.cuda_vbo_resource,
		camera->width, camera->height
	);

	CudaAlogrithm::depthMapTriangulate(
		&planemesh.cuda_vbo_resource,
		&planemesh.cuda_ibo_resource,
		camera->width,
		camera->height,
		planemesh.cudaIndicesCount,
		planeMeshThreshold
	);
}

void CameraGL::saveWrappedResult() {

	std::vector<FrameBuffer> saveType = {
		FrameBuffer::AFTERDISCARD,
		FrameBuffer::COSWEIGHT,
		FrameBuffer::MASK,
		FrameBuffer::MESHNORMAL,
	};
	
	for (auto type : saveType) {
		unsigned char* colorRaw = getFrameBuffer(type)->getRawColorData();
		cv::Mat image(cv::Size(camera->width, camera->height), CV_8UC4, (void*)colorRaw, cv::Mat::AUTO_STEP);
		cv::imwrite(camera->serial + std::to_string(type)+".png", image);
		delete colorRaw;
	}
}

unsigned char* CameraGL::getProcessedColorFrame() {
	unsigned char* colorRaw = new unsigned char[camera->width*camera->height * INPUT_COLOR_CHANNEL];
	glBindTexture(GL_TEXTURE_2D, image);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, colorRaw);
	return colorRaw;
}

void CameraGL::addfloatingSerialGui(glm::mat4 mvp,std::string text) {
	if (!visible)return;

	float windowWidth = (float)ImGui::GetWindowWidth();
	float windowHeight = (float)ImGui::GetWindowHeight();
	ImVec2 p = ImGui::GetCursorScreenPos();

	glm::vec4 worldPos = mvp * camera->modelMat * glm::vec4(0, 0, 0, 1.0);
	worldPos.x /= worldPos.w;
	worldPos.y /= worldPos.w;
	worldPos.z /= worldPos.w;
	ImVec2 sceenPose(windowWidth * (worldPos.x * 0.5 + 0.5), windowHeight * (1.0 - (worldPos.y * 0.5 + 0.5)));
	ImGui::GetWindowDrawList()->AddText(ImVec2(p.x + sceenPose.x, p.y + sceenPose.y), IM_COL32_BLACK, text.c_str());
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
	ImGui::SameLine();
	ImGui::Checkbox(KEY("visible"), &(visible));
	ImGui::ColorEdit3(KEY("color"), (float*)&color); // Edit 3 floats representing a color
	ImGui::SliderFloat(KEY("clip-z"), &camera->farPlane, 0.5f, 15.0f);
	ImGui::SliderFloat(KEY("clip-floor"), &camera->point2floorDistance, -0.1f, 0.2f);
}

// render single realsense mesh
void CameraGL::renderMesh(glm::mat4& mvp, GLuint& program) {
	if (!visible)return;
	auto render = [this, mvp, program](GLuint& vao, int& count) {
		glm::mat4 m = mvp * getModelMat();
		ImguiOpeGL3App::renderElements(m, 0, program, vao, count * 3, GL_FILL);
	};

	planemesh.render(render);
}

void CameraGL::renderFrustum(
	glm::mat4 worldMVP, glm::vec3 camColor,
	GLuint& vao, GLuint& vbo,GLuint render_vertexColor_program,GLuint render_Texture_program
) {
	if (!visible)return;
	glm::mat4 deviceMVP = worldMVP * getModelMat();

	// render camera frustum
	ImguiOpeGL3App::genCameraHelper(
		vao, vbo,
		camera->width, camera->height,
		camera->intri.ppx, camera->intri.ppy, camera->intri.fx, camera->intri.fy,
		camColor, 0.2, false
	);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_CULL_FACE);
	ImguiOpeGL3App::render(deviceMVP, 0, render_vertexColor_program, vao, 3 * 4, GL_TRIANGLES);
	glEnable(GL_CULL_FACE);

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

	glDisable(GL_CULL_FACE);
	ImguiOpeGL3App::render(deviceMVP, 0, render_Texture_program, vao, 3 * 4, GL_TRIANGLES);
	glEnable(GL_CULL_FACE);
}