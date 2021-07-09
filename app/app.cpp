
#include "src/ImguiOpenGL/ImguiOpeGL3App.h"
#include "src/virtualcam/VirtualCam.h"
#include "src/cuda/CudaOpenGLUtils.h"
#include "src/cuda/cudaUtils.cuh"
#include <ctime>
#include "src/json/jsonUtils.h"
#include "src/ExtrinsicCalibrator/ExtrinsicCalibrator.h"
#include "src/CameraManager/CameraManager.h"

#define MAXTEXTURE 16

class RealsenseDepthSythesisApp :public ImguiOpeGL3App {
	GLuint vao, vbo;

	GLuint shader_program, texture_shader_program;
	GLuint project_shader_program;
	GLuint projectTextre_shader_program, screen_projectTextre_shader_program;
	
	VirtualCam* virtualcam;

	CameraManager camManager;

	float t,pointsize=0.1f;

	ExtrinsicCalibrator camPoseCalibrator;

	ImVec4 chromaKeyColor;
	float chromaKeyColorThreshold=2;
	int depthDilationIteration = 0;
	bool autoDepthDilation = true;
	int maskErosion = 0;
	float planeMeshThreshold=1;

	float projectDepthBias = 3e-2;
	bool calculatDeviceWeights=false;

	int curFrame = 0;

public:
	RealsenseDepthSythesisApp():ImguiOpeGL3App(), camManager(){
	}
	~RealsenseDepthSythesisApp() {
		camManager.destory();
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
	}

	void addGui() override {

		if (ImGui::CollapsingHeader("Reconstruct & Texture :")) {
			ImGui::Text("Preprocessing:");
			ImGui::ColorEdit3("chromaKeycolor", (float*)&chromaKeyColor); // Edit 3 floats representing a color
			ImGui::SliderFloat("chromaKeyColorThreshold", &chromaKeyColorThreshold, 0, 5); // Edit 3 floats representing a color
			ImGui::SliderInt("MaskErosion", &maskErosion, 0, 50);
			ImGui::Checkbox("AutoDepthDilation", &autoDepthDilation);
			ImGui::SliderInt("DepthDilation", &depthDilationIteration, 0, 50);

			ImGui::Text("Reconstruct:");
			ImGui::SliderFloat("planeMeshThreshold", &planeMeshThreshold, 0.0f, 90.0f);

			ImGui::Text("Texture Weights:");
			if (ImGui::Button("ResetWeight")) {
				camManager.getProjectTextureDevice([](auto device) {
					device->weight = 1.0;
				});
			}
			ImGui::SameLine();
			ImGui::Checkbox("calculatDeviceWeights", &calculatDeviceWeights);
			camManager.getProjectTextureDevice([](auto device) {
				ImGui::Text("%s - %f", device->camera->serial, device->weight);
			});
			ImGui::SliderFloat("projectDepthBias", &projectDepthBias, 1e-3, 50e-3);
		}
		if (ImGui::CollapsingHeader("Camera Extrinsics")) {
			//aruco calibrate feature point collector params
			camManager.setExtrinsicsUI();
			camPoseCalibrator.addUI();
		}
		if (ImGui::CollapsingHeader("Virtual camera")) {
			virtualcam->addUI();
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
		project_shader_program = GLShader::genShaderProgram(this->window,"projectPointcloud.vs", "projectPointcloud.fs");
		projectTextre_shader_program = GLShader::genShaderProgram(this->window,"vertexcolor.vs","projectTexture.fs");
		screen_projectTextre_shader_program = GLShader::genShaderProgram(this->window,"projectOnScreen.vs","projectTexture.fs");
		
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		virtualcam = new VirtualCam(1280, 720);
		virtualcam->fx = 924.6023559570313;
		virtualcam->fy = 922.5956420898438;
		virtualcam->ppx = 632.439208984375;
		virtualcam->ppy = 356.8707275390625;
	}

	// virtual mesh project depth to real camera (prepared for projective texture)
	void renderScreenViewport(GLuint texture, glm::vec2 offset, glm::vec3 color,float debug = 0, glm::vec2 scale=glm::vec2(0.25,-0.25)) {
		ImguiOpeGL3App::genCameraHelper(
			vao,vbo,
			1, 1, 0.5, 0.5, 0.5, 0.5, glm::vec3(1, 1, 0), 1.0, true
		);

		std::string uniformNames[] = { "color" };
		GLuint texturedepth[] = { texture };
		ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, texturedepth, 1);
		glm::mat4 screenDepthMVP =
			glm::scale(
				glm::translate(
					glm::mat4(1.0),
					glm::vec3(offset.x, offset.y, 0.0)
				),
				glm::vec3(scale.x, scale.y, 1e-3)
			);
		
		std::string outlinerRGB[] = { "outliner_r","outliner_g" ,"outliner_b","debug" };
		float values[] = { color.x,color.y,color.z,debug };
		ImguiOpeGL3App::setUniformFloats(texture_shader_program, outlinerRGB, values, 4);

		ImguiOpeGL3App::render(screenDepthMVP, pointsize, texture_shader_program, vao, 3 * 4, GL_TRIANGLES);
	}

	// render virtual mesh with projective textures
	void renderTexturedMesh(VirtualCam* virtualcam,bool renderOnScreen,bool drawIndex) {
		glm::mat4 mvp = Projection * View * Model;
		// render virtual camera
		glm::mat4 devicemvp = mvp;

		// project textre;
		int deviceIndex = 0;
		GLuint shader = projectTextre_shader_program;

		if (renderOnScreen) {
			shader = screen_projectTextre_shader_program;
			std::string uniformNames[] = {
				"p_w",
				"p_h",
				"p_fx",
				"p_fy",
				"p_ppx",
				"p_ppy",
				"p_near",
				"p_far",
			};
			float values[] = {
				virtualcam->w,
				virtualcam->h,
				virtualcam->fx,
				virtualcam->fy,
				virtualcam->ppx,
				virtualcam->ppy,
				virtualcam->nearplane,
				virtualcam->farplane
			};
			ImguiOpeGL3App::setUniformFloats(shader, uniformNames, values, 8);

			devicemvp = glm::inverse(virtualcam->getModelMat(lookAtPoint, curFrame));
		}

		std::string texNames[MAXTEXTURE];
		GLuint texturs[MAXTEXTURE];

		auto indexTostring = [=](int x) mutable throw() -> std::string
		{
			return std::string("[") + std::to_string(x) + std::string("]");
		};
		int textureCount = camManager.getProjectTextureDevice([
			&texNames, &texturs, &indexTostring, &deviceIndex, &drawIndex, &shader,this
		](auto device) {
			texNames[deviceIndex * 2] = "color" + indexTostring(deviceIndex);
			texNames[deviceIndex * 2 + 1] = "depthtest" + indexTostring(deviceIndex);

			texturs[deviceIndex * 2] = drawIndex ? device->representColorImage : device->image;
			texturs[deviceIndex * 2 + 1] = device->framebuffer.depthBuffer;

			std::string floatnames[] = {
					"w" + indexTostring(deviceIndex),
					"h" + indexTostring(deviceIndex),
					"fx" + indexTostring(deviceIndex),
					"fy" + indexTostring(deviceIndex),
					"ppx" + indexTostring(deviceIndex),
					"ppy" + indexTostring(deviceIndex),
					"near" + indexTostring(deviceIndex),
					"far" + indexTostring(deviceIndex),
					"overlapWeights" + indexTostring(deviceIndex)
			};
			float values[] = {
				device->camera->width,
				device->camera->height,
				device->camera->intri.fx,
				device->camera->intri.fy,
				device->camera->intri.ppx,
				device->camera->intri.ppy,
				0,
				device->camera->farPlane,
				device->weight
			};
			ImguiOpeGL3App::setUniformFloats(shader, floatnames, values, 9);
			deviceIndex++;
		});

		ImguiOpeGL3App::activateTextures(shader, texNames, texturs, deviceIndex * 2);
		std::string indexName[] = { "count","bias"};
		float idx[] = { textureCount, projectDepthBias};
		ImguiOpeGL3App::setUniformFloats(shader, indexName, idx, 2);

		camManager.getFowardDepthWarppingDevice([
			&texNames, &texturs, &indexTostring, &deviceIndex, &drawIndex, &shader, this, &devicemvp
		](auto forwardDepthDevice) {
			deviceIndex = 0;
			camManager.getProjectTextureDevice([
				&texNames, &texturs, &indexTostring, &deviceIndex, &drawIndex, &shader, this, &forwardDepthDevice
			](auto device) {
				glUseProgram(shader);
				GLuint MatrixID = glGetUniformLocation(shader, (std::string("extrinsic") + indexTostring(deviceIndex++)).c_str());
				glm::mat4 model = glm::inverse(device->camera->modelMat) * forwardDepthDevice->camera->modelMat;
				glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &model[0][0]);
				glUseProgram(0);
			});
			forwardDepthDevice->renderMesh(devicemvp, shader);
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
			device->updateImages(chromaKeyColor, chromaKeyColorThreshold, maskErosion, autoDepthDilation, curFrame);
		});

		camManager.getFowardDepthWarppingDevice([this](auto device) {
			device->updateMeshwithCUDA(planeMeshThreshold,depthDilationIteration);
		});		

		//render project depth
		camManager.getProjectTextureDevice([this](auto device) {
			auto renderVirtualMeshProjectDepth = [device, this]() {
				std::string uniformNames[] = {
				"w",
				"h",
				"fx",
				"fy",
				"ppx",
				"ppy",
				"near",
				"far",
				};
				float values[] = {
					device->camera->width,
					device->camera->height,
					device->camera->intri.fx,
					device->camera->intri.fy,
					device->camera->intri.ppx,
					device->camera->intri.ppy,
					0,
					device->camera->farPlane
				};
				ImguiOpeGL3App::setUniformFloats(project_shader_program, uniformNames, values, 8);

				glm::mat4 m = glm::inverse(device->camera->modelMat);
				camManager.getFowardDepthWarppingDevice([this, &m](auto device) {
					device->renderMesh(m, project_shader_program);
				});
			};
			device->framebuffer.render(renderVirtualMeshProjectDepth);
		});

		auto renderTexturedMeshInVirtualview = [this]() {
			renderTexturedMesh(virtualcam, true,false);
		};
		virtualcam->viewport.render(renderTexturedMeshInVirtualview);

		auto renderTexturedIndexMesh = [this]() {
			renderTexturedMesh(virtualcam, true, true);
		};
		virtualcam->debugview.render(renderTexturedIndexMesh);

	}
	
	void render3dworld() {
		glm::mat4 mvp = Projection * View * Model;

		// render center axis
		ImguiOpeGL3App::genOrigionAxis(vao, vbo);
		glm::mat4 mvpAxis = Projection * View * glm::translate(glm::mat4(1.0), lookAtPoint) * Model;
		ImguiOpeGL3App::render(mvpAxis, pointsize, shader_program, vao, 6, GL_LINES);

		std::vector<glm::vec2> windows = {
			glm::vec2(0.25,0.75),
			glm::vec2(0.75,0.75),
		};
		int deviceIndex = 0;
		camManager.getAllDevice([this, &mvp, &windows, &deviceIndex](auto device, auto allDevice) {
			renderFrustum(device);
			//renderScreenViewport(device->framebuffer.depthBuffer, windows[deviceIndex++], glm::vec3(device->color.x, device->color.y, device->color.z));
			//device->renderMesh(mvp, shader_program);
			camPoseCalibrator.waitCalibrateCamera(device, allDevice);
		});

		// render virtual camera frustum
		glm::mat4 devicemvp = mvp * virtualcam->getModelMat(lookAtPoint, curFrame);
		virtualcam->renderFrustum(devicemvp, vao, vbo, shader_program);

		renderTexturedMesh(virtualcam, false, false);
		camPoseCalibrator.render(mvp, shader_program);
	}

	void mainloop() override {
		if (calculatDeviceWeights) {
			// calculate weight depend on camera position
			glm::mat4 vmodelMat = virtualcam->getModelMat(lookAtPoint, curFrame);
			camManager.updateProjectTextureWeight(vmodelMat);
		}

		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w/2, display_h/2);
		render3dworld();
		glViewport(0, 0, display_w, display_h);

		renderScreenViewport(virtualcam->viewport.texColorBuffer, glm::vec2(0.5, 0.5), virtualcam->color,0, glm::vec2(0.5, -0.5));
		renderScreenViewport(virtualcam->viewport.depthBuffer, glm::vec2(0.5, -0.5), virtualcam->color,0,glm::vec2(0.5,-0.5));

		camManager.getSingleDebugDevice([this](auto cam) {
			renderScreenViewport(cam.image, glm::vec2(-0.25, 0.25), virtualcam->color,1.0);
			renderScreenViewport(cam.depthvis, glm::vec2(-0.25, 0.75), virtualcam->color,1.0);
		});

		curFrame++;
	}
};

int main() {
	RealsenseDepthSythesisApp viewer;
	viewer.initImguiOpenGL3();
}