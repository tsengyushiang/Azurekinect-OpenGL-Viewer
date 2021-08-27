
#include "src/ImguiOpenGL/ImguiOpeGL3App.h"
#include "src/virtualcam/VirtualCam.h"
#include "src/cuda/CudaOpenGLUtils.h"
#include "src/cuda/cudaUtils.cuh"
#include <ctime>
#include "src/json/jsonUtils.h"
#include "src/ExtrinsicCalibrator/ExtrinsicCalibrator.h"
#include "src/CameraManager/CameraManager.h"
#include "src/virtualcam/VirtualRouteAnimator.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <filesystem>

class FowardWarppingApp :public ImguiOpeGL3App {
	GLuint vao, vbo;

	GLuint shader_program;

	GLuint texture_shader_program;
	GLuint screen_texturedMesh_shader_program;

	GLuint screen_MeshMask_shader_program;
	GLuint screen_facenormal_shader_program;
	GLuint screen_cosWeight_shader_program;
	GLuint screen_cosWeightDiscard_shader_program;
	GLuint cosWeightDiscard_shader_program;

	VirtualCam* virtualcam;
	VirtualRouteAnimator animator;
	int currentRecordFrame = -1;

	CameraManager camManager;

	float t,pointsize=0.1f;

	ExtrinsicCalibrator camPoseCalibrator;

	ImVec4 chromaKeyColor;
	float chromaKeyColorThreshold=2;
	int pointsSmoothing = 10;
	bool autoDepthDilation = false;
	int maskErosion = 3;
	float planeMeshThreshold=0;
	float cullTriangleThreshold = 0.25;

	float projectDepthBias = 3e-2;
	bool calculatDeviceWeights=false;

	int curFrame = 0;

public:
	FowardWarppingApp():ImguiOpeGL3App(), camManager(){
		//OpenCVUtils::saveMarkerBoard();
	}
	~FowardWarppingApp() {
		camManager.destory();
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
	}
	void onBeforeRender() override {
		//run animation if is triggered
		currentRecordFrame = animator.animeVirtualCamPose(virtualcam->pose);
		if (currentRecordFrame > -1) {
			std::cout << currentRecordFrame << std::endl;
		}
	}
	void onAfterRender() override {
		if (currentRecordFrame > -1) {
			// save all animation frames
			std::string outputFolder = cv::utils::fs::join("./", std::to_string(currentRecordFrame));
			cv::utils::fs::createDirectory(outputFolder);
			camManager.getFoward3DWrappingDevice([this,&outputFolder](auto cam) {
				std::string imgefilename = cv::utils::fs::join(outputFolder, cam->camera->serial + ".png");
				unsigned char* colorRaw = cam->getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->getRawColorData();				
				cv::Mat image(cv::Size(cam->camera->width, cam->camera->height), CV_8UC4, (void*)colorRaw, cv::Mat::AUTO_STEP);
				cv::imwrite(imgefilename, image);
				currentRecordFrame = -1;
			});
		}

		camManager.recordFrame();
	}
	void addGui() override {
		// input cameras	
		{
			ImGui::Begin("Input Cameras");
			{
				// Using a Child allow to fill all the space of the window.
				// It also alows customization
				ImGui::BeginChild("Textures");
				// Get the size of the child (i.e. the whole draw size of the windows).
				ImVec2 wsize = ImGui::GetWindowSize();
				float inputCount = camManager.size();
				wsize.x /= inputCount;
				wsize.y /= 2;
				// Because I use the texture from OpenGL, I need to invert the V from the UV.
				camManager.getAllDevice([&wsize](auto cam) {
					ImGui::Image((ImTextureID)cam->depthvis, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					ImGui::SameLine();
					});
				ImGui::Text("Depth");

				camManager.getAllDevice([&wsize](auto cam) {
					ImGui::Image((ImTextureID)cam->image, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					ImGui::SameLine();
					});
				ImGui::Text("Color");

				ImGui::EndChild();
			}
			ImGui::End();
		}

		// output warpping result and meta data
		{
			ImGui::Begin("Warpping Result");
			{
				// Using a Child allow to fill all the space of the window.
				// It also alows customization
				ImGui::BeginChild("Textures");
				// Get the size of the child (i.e. the whole draw size of the windows).
				ImVec2 wsize = ImGui::GetWindowSize();
				float inputCount = camManager.size();
				wsize.y /= 2;
				camManager.getOutputDebugDevice([this, &wsize](auto cam) {
					ImGui::Image((ImTextureID)cam.getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->texColorBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
				});
				wsize.x /= 2;
				wsize.y /= 2;
				camManager.getOutputDebugDevice([this,&wsize](auto cam) {
					ImGui::Image((ImTextureID)cam.getFrameBuffer(CameraGL::FrameBuffer::MASK)->texColorBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					ImGui::SameLine();
					ImGui::Image((ImTextureID)cam.getFrameBuffer(CameraGL::FrameBuffer::COSWEIGHT)->texColorBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));

					ImGui::Image((ImTextureID)cam.getFrameBuffer(CameraGL::FrameBuffer::MESHNORMAL)->depthBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					ImGui::SameLine();
					ImGui::Image((ImTextureID)cam.getFrameBuffer(CameraGL::FrameBuffer::MESHNORMAL)->texColorBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
				});
				ImGui::EndChild();
			}
			ImGui::End();
		}

	}
	void addMenu() override {

		if (ImGui::CollapsingHeader("Reconstruct & Texture :")) {
			ImGui::Text("Preprocessing:");
			ImGui::ColorEdit3("chromaKeycolor", (float*)&chromaKeyColor); // Edit 3 floats representing a color
			ImGui::SliderFloat("chromaKeyColorThreshold", &chromaKeyColorThreshold, 0, 5); // Edit 3 floats representing a color
			ImGui::SliderInt("MaskErosion", &maskErosion, 0, 50);
			ImGui::Checkbox("AutoDepthDilation", &autoDepthDilation);
			ImGui::SliderInt("pointsSmoothing", &pointsSmoothing, 0, 50);

			ImGui::Text("Reconstruct:");
			ImGui::SliderFloat("planeMeshThreshold", &planeMeshThreshold, 0.0f, 90.0f);
			ImGui::SliderFloat("cullTriangleThreshold", &cullTriangleThreshold, 0, 1);

		}
		if (ImGui::CollapsingHeader("Camera Extrinsics Calibrator")) {			
			//aruco calibrate feature point collector params
			camPoseCalibrator.addUI();
		}
		if (ImGui::CollapsingHeader("Virtual camera")) {
			virtualcam->addUI();
			animator.addUI(virtualcam->pose);
		}
		if (ImGui::CollapsingHeader("Local File Manager")) {
			camManager.setExtrinsicsUI();
			camManager.addLocalFileUI();
		}
		if (ImGui::CollapsingHeader("Cameras Manager")) {
			camManager.addCameraUI();
		}
		if (ImGui::CollapsingHeader("Debug Option")) {
			camManager.addDepthAndTextureControlsUI();

			static char url[25] = "virtual-view-color.png";
			ImGui::InputText("##urlInput", url, 20);
			ImGui::SameLine();
			if (ImGui::Button("save virutal view color")) {
				unsigned char* colorRaw = virtualcam->viewport.getRawColorData();
				cv::Mat image(cv::Size(virtualcam->w, virtualcam->h), CV_8UC4, (void*)colorRaw, cv::Mat::AUTO_STEP);
				cv::imwrite(url, image);
				delete colorRaw;
			}
		}
	}
	void initGL() override {
		shader_program = GLShader::genShaderProgram(this->window, "vertexcolor.vs", "vertexcolor.fs");
		texture_shader_program = GLShader::genShaderProgram(this->window, "texture.vs", "texture.fs");
		
		screen_texturedMesh_shader_program = GLShader::genShaderProgram(this->window, "projectOnScreen.vs", "texture.fs");
		screen_facenormal_shader_program = GLShader::genShaderProgram(this->window, "projectOnScreen.vs", "facenormal.fs");
		screen_cosWeight_shader_program = GLShader::genShaderProgram(this->window, "projectOnScreen.vs", "cosWeight.fs");
		screen_cosWeightDiscard_shader_program = GLShader::genShaderProgram(this->window, "projectOnScreen.vs", "cosWeightDiscardwTexture.fs");
		cosWeightDiscard_shader_program = GLShader::genShaderProgram(this->window, "vertexcolor.vs", "cosWeightDiscardwTexture.fs");
		screen_MeshMask_shader_program = GLShader::genShaderProgram(this->window, "projectOnScreen.vs", "mask.fs");

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		//virtualcam = new VirtualCam(1280, 720);
		//virtualcam->fx = 924.6023559570313;
		//virtualcam->fy = 922.5956420898438;
		//virtualcam->ppx = 632.439208984375;
		//virtualcam->ppy = 356.8707275390625;

		virtualcam = new VirtualCam(1920, 1080);
		virtualcam->fx = 913.4943237304688;
		virtualcam->fy = 913.079833984375;
		virtualcam->ppx = 960.0040283203125;
		virtualcam->ppy = 552.7597045898438;
	}

	void updateForwardWrappingTexture(GLuint shader, VirtualCam* virtualcam, bool drawIndex,CameraGL::FrameBuffer type) {
		glm::mat4 devicemvp = glm::inverse(virtualcam->getModelMat(lookAtPoint, curFrame));
		std::string uniformNames[] = {
			"p_w",
			"p_h",
			"p_fx",
			"p_fy",
			"p_ppx",
			"p_ppy",
			"p_near",
			"p_far",
			"weightThreshold"
		};
		float values[] = {
			virtualcam->w,
			virtualcam->h,
			virtualcam->fx,
			virtualcam->fy,
			virtualcam->ppx,
			virtualcam->ppy,
			virtualcam->nearplane,
			virtualcam->farplane,
			cullTriangleThreshold
		};
		ImguiOpeGL3App::setUniformFloats(shader, uniformNames, values, 8+1);

		camManager.getFoward3DWrappingDevice([
			&type, &drawIndex, &shader, this, &devicemvp
		](auto forwardwrappingDevice) {
			std::string uniformNames[] = { "color" };
			GLuint texture[] = { drawIndex ? forwardwrappingDevice->representColorImage : forwardwrappingDevice->image};
			ImguiOpeGL3App::activateTextures(shader, uniformNames, texture, 1);
			auto render = [&forwardwrappingDevice, &shader, this, &devicemvp]() {
				ImguiOpeGL3App::setUniformMat(shader, "modelMat", forwardwrappingDevice->camera->modelMat);
				forwardwrappingDevice->renderMesh(devicemvp, shader);
			};
			forwardwrappingDevice->getFrameBuffer(type)->render(render);
		});
	}	
	
	// render single realsense camera pose and color image in world coordinate
	void renderFrustum(std::vector<CameraGL>::iterator device) {
		glm::mat4 mvp = Projection * View * Model;

		glm::vec3 camColor = glm::vec3(device->color.x, device->color.y, device->color.z);
		bool isCalibratingCamera = camPoseCalibrator.checkIsCalibrating(device->camera->serial, camColor);
		device->renderFrustum(
			mvp, camColor,vao,vbo, shader_program, texture_shader_program
		);
	}
	
	// render realsense mesh on framebuffer
	void framebufferRender() override {
		camManager.deleteIdleCam();

		camManager.getAllDevice([this](auto device) {
			device->updateImages(chromaKeyColor, chromaKeyColorThreshold, maskErosion, autoDepthDilation);
		});

		camManager.getFoward3DWrappingDevice([this](auto device) {
			device->updateMeshwithCUDA(planeMeshThreshold,pointsSmoothing);
		});

		updateForwardWrappingTexture(screen_MeshMask_shader_program,virtualcam,false,CameraGL::FrameBuffer::MASK);
		updateForwardWrappingTexture(screen_facenormal_shader_program,virtualcam,false, CameraGL::FrameBuffer::MESHNORMAL);
		updateForwardWrappingTexture(screen_cosWeight_shader_program,virtualcam,false, CameraGL::FrameBuffer::COSWEIGHT);
		updateForwardWrappingTexture(screen_cosWeightDiscard_shader_program,virtualcam,false, CameraGL::FrameBuffer::AFTERDISCARD);	

	}
	
	void render3dworld() {
		glm::mat4 mvp = Projection * View * Model;

		// render center axis
		ImguiOpeGL3App::genOrigionAxis(vao, vbo);
		glm::mat4 mvpAxis = Projection * View * glm::translate(glm::mat4(1.0), lookAtPoint) * Model;
		ImguiOpeGL3App::render(mvpAxis, pointsize, shader_program, vao, 6, GL_LINES);

		camManager.getAllDevice([this, &mvp](auto device, auto allDevice) {
			renderFrustum(device);
			camPoseCalibrator.waitCalibrateCamera(device, allDevice);
		});

		// render virtual camera frustum
		glm::mat4 devicemvp = mvp * virtualcam->getModelMat(lookAtPoint, curFrame);
		virtualcam->renderFrustum(devicemvp, vao, vbo, shader_program);

		GLuint _3dshader = cosWeightDiscard_shader_program;
		camManager.getFoward3DWrappingDevice([&_3dshader, &mvp, this](auto device) {
			std::string uniformName[] = {"weightThreshold"};
			float values[] = {cullTriangleThreshold};
			ImguiOpeGL3App::setUniformFloats(_3dshader, uniformName, values, 1);
			std::string uniformNames[] = { "color" };
			GLuint texture[] = { device->image };
			ImguiOpeGL3App::activateTextures(_3dshader, uniformNames, texture, 1);
			ImguiOpeGL3App::setUniformMat(_3dshader, "modelMat", device->camera->modelMat);
			device->renderMesh(mvp, _3dshader);
		});
		camPoseCalibrator.render(mvp, shader_program);
	}

	void mainloop() override {
		if (calculatDeviceWeights) {
			// calculate weight depend on camera position
			glm::mat4 vmodelMat = virtualcam->getModelMat(lookAtPoint, curFrame);
			camManager.updateProjectTextureWeight(vmodelMat);
		}
		render3dworld();
		curFrame++;
	}
};

int main() {
	FowardWarppingApp viewer;
	viewer.initImguiOpenGL3();
}