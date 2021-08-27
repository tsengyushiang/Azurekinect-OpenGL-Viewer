#pragma once

#include "../ImguiOpenGL/ImguiOpeGL3App.h"
#include "../InputCamera/InputBase.h"
#include "../cuda/CudaOpenGLUtils.h"
#include "../cuda/cudaUtils.cuh"
#include "../json/jsonUtils.h"

#define CamIterator std::vector<CameraGL>::iterator&

class CameraGL {
	
	GLFrameBuffer maskInVirtualView;
	GLFrameBuffer meshnormalInVirtualView;
	GLFrameBuffer cosWeightInVirtualView;
	GLFrameBuffer afterDicardInVirtualView;

public:
	
	enum FrameBuffer
	{
		MASK, MESHNORMAL, COSWEIGHT, AFTERDISCARD
	};
	GLFrameBuffer* getFrameBuffer(FrameBuffer);

	bool useDepth = true;
	bool useTexture = true;
	glm::mat4 getModelMat();

	bool ready2Delete = false;
	InputBase* camera;

	ImVec4 color;

	GLuint image, depthvis, representColorImage;
	cudaGraphicsResource_t image_cuda,depthvis_cuda, representColorImage_cuda;

	CudaGLDepth2PlaneMesh planemesh;

	// project texture weight
	float weight = 1.0;
	bool create3d = true;
	CameraGL(InputBase* cam);
	void destory();
	unsigned char* getProcessedColorFrame();
	void saveWrappedResult();
	void addui();
	void updateImages(ImVec4 chromaKeyColor, float chromaKeyColorThreshold,
		int maskErosionSize, bool autoDepthDilation);
	// pass realsense data to cuda and compute plane mesh and point cloud
	void updateMeshwithCUDA(float planeMeshThreshold, int pointSmoothing);
	
	// render single realsense mesh
	void renderMesh(glm::mat4& mvp, GLuint& program);
	
	void renderFrustum(
		glm::mat4 worldMVP, glm::vec3 camColor,
		GLuint& vao, GLuint& vbo, GLuint render_vertexColor_program, GLuint render_Texture_program );
};