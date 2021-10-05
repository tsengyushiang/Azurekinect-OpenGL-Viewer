
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
#include "src/ImguiOpenGL/ImGuiTransformControl.h"
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

	/*
	* clipping bounding box, use matrix for transform controls, only translateXYZ is used.
	*/
	glm::vec3 clippingBoundingBoxMin = glm::vec3(-0.5, -0.1, -0.5);
	glm::vec3 clippingBoundingBoxMax = glm::vec3(0.5, 1, 0.5);
	glm::mat4 clippingBoundingBoxMat4 = glm::scale(glm::mat4(1.0),glm::vec3(1,2,1));

	TransformContorl transformWidget;

	int nearsetK = 2;
	VirtualRouteAnimator animator;

	VirtualCam* virtualcam;

	// snapshot warpping result
	RecordProgress currentRecordFrame = {-1,-1};

	CameraManager camManager;

	float t,pointsize=0.1f;

	ExtrinsicCalibrator camPoseCalibrator;

	bool use3D = false;
	// for better performance when calibration
	bool preprocessing = false;
	bool warpping = false;
	bool warppingShowDepth = false;

	bool detectMarkerEveryFrame = false;
	ImVec4 chromaKeyColor = ImVec4(84.0/255,64.0/255,109.0/255,1.0);
	glm::vec3 chromaKeyColorThreshold;
	int pointsSmoothing = 10;
	bool autoDepthDilation = false;
	int maskErosion = 3;
	float planeMeshThreshold=0;
	float cullTriangleThreshold = 0.25;

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
		if (currentRecordFrame.frameIndex > -1) {
			camManager.getNearestKcamera(nearsetK, virtualcam->getModelMat(lookAtPoint, curFrame), lookAtPoint, [this](auto cam) {
				cam->camera->setFrameIndex(currentRecordFrame.frameIndex);
			});
		}
	}
	void onAfterRender() override {
		if (currentRecordFrame.frameIndex > -1) {

			// save all animation frames
			std::ostringstream sstr;
			sstr << std::setfill('0') << std::setw(8) <<currentRecordFrame.frameIndex;
			std::string currentRecordfolder = sstr.str();

			std::string outputFolder = cv::utils::fs::join(animator.folder, currentRecordfolder);
			cv::utils::fs::createDirectory(outputFolder);

			int i = 0;
			camManager.getNearestKcamera(nearsetK, virtualcam->getModelMat(lookAtPoint, curFrame), lookAtPoint, [&i,this, &outputFolder](auto cam) {
				std::ostringstream sstr;
				sstr << std::setfill('0') << std::setw(8) << i++ << ".png";
				std::string knearestview = sstr.str();
				std::string imgefilename = cv::utils::fs::join(outputFolder, knearestview);

				if(warppingShowDepth)
				{
					// get gpu depth frame
					float* depthRaw = cam->getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->getRawDepthData();
					cv::Mat image(cv::Size(cam->camera->width, cam->camera->height), CV_8UC4);
					for (int i = 0; i < cam->camera->height; i++) {
						for (int j = 0; j < cam->camera->width; j++) {
							int index = i * cam->camera->width + j;
							float depthvalue = depthRaw[index];
								image.at < cv::Vec4b >(i,j) = cv::Vec4b(
									depthvalue*255,
									depthvalue*255,
									depthvalue*255,
									depthvalue > 0.999?0:255
								);
						}
					}
					cv::imwrite(imgefilename, image);
					delete depthRaw;
				}
				else {
					// get gpu color frame
					unsigned char* colorRaw = cam->getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->getRawColorData();
					cv::Mat image(cv::Size(cam->camera->width, cam->camera->height), CV_8UC4, (void*)colorRaw, cv::Mat::AUTO_STEP);
					cv::imwrite(imgefilename, image);
					delete colorRaw;
				}

				currentRecordFrame = {-1,-1};
			});
		}

		camManager.recordFrame();
	}
	bool addOpenGLPanelGui() override{

		if (ImGui::Button("Snapshot OpenGL")) {
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			std::ostringstream sstr;
			sstr << "./Opengl-snapshot" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S")<<".png";
			std::string imgefilename = sstr.str();

			std::cout << "save opengl" << std::endl;
			unsigned char* colorRaw = main->getRawColorData();
			cv::Mat image(cv::Size(width, height), CV_8UC4, (void*)colorRaw, cv::Mat::AUTO_STEP);
			cv::flip(image, image, 0);
			cv::imwrite(imgefilename, image);
		}

		camManager.getAllDevice([this](auto device, auto allDevice) {
			device->addfloatingSerialGui(Projection * View * Model, device->camera->serial);
		});

		bool isUsing = transformWidget.addWidget(View, Projection);
		return isUsing;
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
				ImGui::Checkbox("Warpping", &warpping);
				ImGui::Checkbox("Depth/Color", &warppingShowDepth);
				// Using a Child allow to fill all the space of the window.
				// It also alows customization
				ImGui::BeginChild("Textures");
				// Get the size of the child (i.e. the whole draw size of the windows).
				ImVec2 wsize = ImGui::GetWindowSize();
				float inputCount = camManager.size();
				wsize.y /= inputCount;

				camManager.getAllDevice([&wsize, this](auto cam) {
					if (warppingShowDepth) {
						ImGui::Image((ImTextureID)cam->getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->depthBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					}
					else {
						ImGui::Image((ImTextureID)cam->getFrameBuffer(CameraGL::FrameBuffer::AFTERDISCARD)->texColorBuffer, ImVec2(wsize.x, wsize.y), ImVec2(0, 1), ImVec2(1, 0));
					}
				});

				ImGui::EndChild();
			}
			ImGui::End();
		}


	}
	void addMenu() override {

		if (ImGui::CollapsingHeader("Reconstruct & Texture :")) {

			if (ImGui::Button("Clipping BoundingBox")) {
				transformWidget.attachMatrix4(clippingBoundingBoxMat4);
			}
			transformWidget.addMenu();

			ImGui::ColorEdit3("chromaKeycolor", (float*)&chromaKeyColor); // Edit 3 floats representing a color
			ImGui::SliderFloat("H-Threshold", &chromaKeyColorThreshold.x, 0, 255); // Edit 3 floats representing a color
			ImGui::SliderFloat("S-Threshold", &chromaKeyColorThreshold.y, 0, 255); // Edit 3 floats representing a color
			ImGui::SliderFloat("V-Threshold", &chromaKeyColorThreshold.z, 0, 255); // Edit 3 floats representing a color

			ImGui::Checkbox("Preprocessing :", &preprocessing);
			ImGui::SliderInt("MaskErosion", &maskErosion, 0, 50);
			ImGui::Checkbox("AutoDepthDilation", &autoDepthDilation);

			ImGui::Checkbox("Reconstruct:", &use3D);
			ImGui::SliderFloat("planeMeshThreshold", &planeMeshThreshold, 0.0f, 90.0f);
			ImGui::SliderFloat("cullTriangleThreshold", &cullTriangleThreshold, 0, 1);
			ImGui::SliderInt("pointsSmoothing", &pointsSmoothing, 0, 50);

		}
		if (ImGui::CollapsingHeader("Floor,Extrinsics Calibrator")) {			
			//aruco calibrate feature point collector params
			if (ImGui::Button("Calibrated floor by region")) {
				camManager.getAllDevice([this](auto device, auto allDevice) {
					camPoseCalibrator.fitMarkersOnEstimatePlane(device->camera, OpenCVUtils::getArucoMarkerConvexRegion);
				});
			}

			ImGui::Text("Cam-to-cam extrinsics calibrate");
			camPoseCalibrator.addUI();
			{
				ImGui::SliderInt("maxCollectFrame", &camPoseCalibrator.maxCollectFeaturePoint, 1, 144);
				ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
				ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
				ImGui::PopItemFlag();
				float progress = float(camPoseCalibrator.alreadyGet) / camPoseCalibrator.maxCollectFeaturePoint;
				ImGui::SliderFloat("CollectProgress", &progress, 0, 1);
				ImGui::PopStyleVar();
			}
			if (ImGui::Button("Start/Capture")) {
				bool noCalibratedCam = true;
				camManager.getAllDevice([this,&noCalibratedCam](auto device, auto allDevice) {
					if (device->camera->calibrated)noCalibratedCam = false;
				});

				if (noCalibratedCam) {
					camManager.getAllDevice([this, &noCalibratedCam](auto device, auto allDevice) {
						if (allDevice.begin()->camera->serial == device->camera->serial) {
							device->camera->calibrated = true;
						}
					});
				}
				camPoseCalibrator.startCollectPoint = true;
			}
		}
		if (ImGui::CollapsingHeader("Manual tune Camera pose")) {

			ImGui::Text("Pick 2 for ICP:");
			camManager.addManulPick2Gui();
			if (ImGui::Button("RunICP##manualPicks")) {
				auto mat = camPoseCalibrator.runIcp(
					camManager.pickedCams.source, camManager.pickedCams.target, camPoseCalibrator.uniformDepthSample);
				camManager.pickedCams.source->modelMat = mat * camManager.pickedCams.source->modelMat;
			}

			ImGui::Text("Transform control:");
			camManager.getAllDevice([this](auto device) {
				if(ImGui::Button((device->camera->serial+ "##manualDragPos").c_str())) {
					transformWidget.attachMatrix4(device->camera->modelMat);
				}
			});
			transformWidget.addMenu();

			ImGui::Text("Manual Adjust hardware:");
			camManager.getAllDevice([this](auto device) {
				ImGui::Checkbox((device->camera->serial + "##showOpenCV").c_str(),&device->camera->showOpenCVwindow);
			});
		}
		if (ImGui::CollapsingHeader("Virtual camera")) {
			virtualcam->addUI();
			animator.addUI(virtualcam->pose);
			ImGui::SliderInt("Nearest K WarppingResult", &nearsetK, 0, camManager.size());
		}
		if (ImGui::CollapsingHeader("LocalFiles")) {
			camManager.setExtrinsicsUI();
			camManager.addLocalFileUI();
		}
		if (ImGui::CollapsingHeader("Recorder")) {
			camManager.addRecorderUI();
		}
		if (ImGui::CollapsingHeader("Cameras")) {
			camManager.addCameraUI();
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

		//  init pose for real world catpure
		lookAtPoint.y = 1.0;
		virtualcam->pose.AzimuthAngle = -1.564;
		virtualcam->pose.PolarAngle = 1.570;
		virtualcam->pose.distance = 2;
		virtualcam->farplane = 3;		
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
			device->updateImages(chromaKeyColor, chromaKeyColorThreshold,
				glm::inverse(clippingBoundingBoxMat4), clippingBoundingBoxMax, clippingBoundingBoxMin);
			if(preprocessing)
				device->imagesPreprocessing(maskErosion, autoDepthDilation);
		});

		if (use3D) {
			camManager.getFoward3DWrappingDevice([this](auto device) {
				device->updateMeshwithCUDA(planeMeshThreshold, pointsSmoothing);
				});

			if (warpping) {
				updateForwardWrappingTexture(screen_MeshMask_shader_program, virtualcam, false, CameraGL::FrameBuffer::MASK);
				updateForwardWrappingTexture(screen_facenormal_shader_program, virtualcam, false, CameraGL::FrameBuffer::MESHNORMAL);
				updateForwardWrappingTexture(screen_cosWeight_shader_program, virtualcam, false, CameraGL::FrameBuffer::COSWEIGHT);
				updateForwardWrappingTexture(screen_cosWeightDiscard_shader_program, virtualcam, false, CameraGL::FrameBuffer::AFTERDISCARD);
			}
		}
	}
	
	void render3dworld() {
		glm::mat4 mvp = Projection * View * Model;

		// render center axis
		ImguiOpeGL3App::genOrigionAxis(vao, vbo);
		glm::mat4 mvpAxis = Projection * View * glm::translate(glm::mat4(1.0), lookAtPoint) * Model;
		ImguiOpeGL3App::render(mvpAxis, pointsize, shader_program, vao, 6, GL_LINES);
		
		// render clipp bounding obx
		ImguiOpeGL3App::genBoundingboxWireframe(vao, vbo, 
			clippingBoundingBoxMin,
			clippingBoundingBoxMax
		);
		glm::mat4 mvpBoundingbox = Projection * View * Model * clippingBoundingBoxMat4;
		ImguiOpeGL3App::render(mvpBoundingbox, pointsize, shader_program, vao, 24, GL_LINES);

		camManager.getAllDevice([this, &mvp](auto device, auto allDevice) {
			renderFrustum(device);
		});

		// render virtual camera frustum
		glm::mat4 devicemvp = mvp * virtualcam->getModelMat(lookAtPoint, curFrame);
		virtualcam->renderFrustum(devicemvp, vao, vbo, shader_program);

		if (use3D) {
			GLuint _3dshader = cosWeightDiscard_shader_program;
			camManager.getFoward3DWrappingDevice([&_3dshader, &mvp, this](auto device) {
				std::string uniformName[] = { "weightThreshold" };
				float values[] = { cullTriangleThreshold };
				ImguiOpeGL3App::setUniformFloats(_3dshader, uniformName, values, 1);
				std::string uniformNames[] = { "color" };
				GLuint texture[] = { device->image };
				ImguiOpeGL3App::activateTextures(_3dshader, uniformNames, texture, 1);
				ImguiOpeGL3App::setUniformMat(_3dshader, "modelMat", device->camera->modelMat);
				device->renderMesh(mvp, _3dshader);
				});
		}

		camPoseCalibrator.render(mvp, shader_program);
	}

	void mainloop() override {
		if (calculatDeviceWeights) {
			// calculate weight depend on camera position
			glm::mat4 vmodelMat = virtualcam->getModelMat(lookAtPoint, curFrame);
			camManager.updateProjectTextureWeight(vmodelMat);
		}
		
		camManager.getAllDevice([this](auto device, auto allDevice) {
			camPoseCalibrator.waitCalibrateCamera(device, allDevice);
		});

		render3dworld();

		curFrame++;
	}
};

int main() {
	FowardWarppingApp viewer;
	viewer.initImguiOpenGL3();
}