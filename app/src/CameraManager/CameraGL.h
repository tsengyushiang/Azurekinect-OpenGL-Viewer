#pragma once

#include "../imgui/ImguiOpeGL3App.h"
#include "../realsnese//RealsenseDevice.h"
#include "../cuda/CudaOpenGLUtils.h"
#include "../cuda/cudaUtils.cuh"
#include "../json/jsonUtils.h"

class CameraGL {
public:
	bool use2createMesh = true;
	bool ready2Delete = false;
	RealsenseDevice* camera;

	ImVec4 color;

	GLuint image,representColorImage;
	cudaGraphicsResource_t image_cuda, representColorImage_cuda;

	CudaGLDepth2PlaneMesh planemesh;

	// project texture weight
	float weight = 1.0;

	// project depthbuffer
	GLFrameBuffer framebuffer;

	CameraGL();
	void destory();
	void save();
	void addui();
	void updateImages(ImVec4 chromaKeyColor, float chromaKeyColorThreshold, int maskErosionSize, bool autoDepthDilation, int curFram);
	// pass realsense data to cuda and compute plane mesh and point cloud
	void updateMeshwithCUDA(float planeMeshThreshold, int depthDilationIterationCounte);
	
	// render single realsense mesh
	void renderMesh(glm::mat4& mvp, GLuint& program);
	
	void renderFrustum(
		glm::mat4 worldMVP, glm::vec3 camColor,
		GLuint& vao, GLuint& vbo, GLuint render_vertexColor_program, GLuint render_Texture_program );
};