
#include "src/imgui/ImguiOpeGL3App.h"
#include "src/realsnese//RealsenseDevice.h"
#include "src/opencv/opecv-utils.h"
#include "src/pcl/examples-pcl.h"
#include "src/cuda/CudaOpenGLUtils.h"
#include "src/cuda/cudaUtils.cuh"
#include <ctime>

class CorrespondPointCollector {

public :
	GLuint srcVao, srcVbo;
	GLuint trgVao, trgVbo;

	RealsenseDevice* sourcecam;
	RealsenseDevice* targetcam;
	int vaildCount = 0;
	int size;
	float pushThresholdmin = 0.1f;

	std::vector<glm::vec3> srcPoint;
	std::vector<glm::vec3> dstPoint;
	float* source;
	float* target;
	float* result;
	CorrespondPointCollector(RealsenseDevice* srcCam, RealsenseDevice* trgCam,int count = 10,float threshold=0.2f) {
		sourcecam = srcCam;
		targetcam = trgCam;
		size = count;
		pushThresholdmin = threshold;
		source = (float*)calloc(size * 3 * 2, sizeof(float));
		target = (float*)calloc(size * 3 * 2, sizeof(float));
		result = (float*)calloc(size * 3 * 2, sizeof(float));
		glGenVertexArrays(1, &srcVao);
		glGenBuffers(1, &srcVbo);
		glGenVertexArrays(1, &trgVao);
		glGenBuffers(1, &trgVbo);

		//sourcecam->opencvImshow = true;
		//targetcam->opencvImshow = true;
	}
	~CorrespondPointCollector() {
		free(source);
		free(target);
		free(result);
		glDeleteVertexArrays(1, &srcVao);
		glDeleteBuffers(1, &srcVbo);
		glDeleteVertexArrays(1, &trgVao);
		glDeleteBuffers(1, &trgVbo);

		//sourcecam->opencvImshow = false;
		//targetcam->opencvImshow = false;
	}

	void render(glm::mat4 mvp,GLuint shader_program) {
		ImguiOpeGL3App::setPointsVAO(srcVao, srcVao, source, size);
		ImguiOpeGL3App::setPointsVAO(trgVao,trgVbo, target, size);

		ImguiOpeGL3App::render(mvp, 10, shader_program, srcVao, size, GL_POINTS);
		ImguiOpeGL3App::render(mvp, 10, shader_program,trgVao, size, GL_POINTS);
	}

	bool pushCorrepondPoint(glm::vec3 src, glm::vec3 trg) {

		int index = vaildCount;
		if (index != 0) {
			// check threshold
			glm::vec3 p;
			int previousIndex = index - 1;

			for (auto p : srcPoint) {
				auto d = glm::length(p - src);
				if (d < pushThresholdmin) {
					return false;
				}
			}

			for (auto p : dstPoint) {
				auto d = glm::length(p - trg);
				if (d < pushThresholdmin) {
					return false;
				}
			}
		}

		srcPoint.push_back(src);
		dstPoint.push_back(trg);

		source[index*6+0] = src.x;
		source[index*6+1] = src.y;
		source[index*6+2] = src.z;
		source[index*6+3]=1.0;
		source[index*6+4]=0.0;
		source[index*6+5]=0.0;
		
		target[index*6+0] = trg.x;
		target[index*6+1] = trg.y;
		target[index*6+2] = trg.z;
		target[index*6+3]=0.0;
		target[index*6+4]=1.0;
		target[index*6+5]=0.0;
		
		vaildCount++;
		
		return true;
	}

	glm::mat4 calibrate() {
		glm::mat4 transform = pcl_pointset_rigid_calibrate(size, srcPoint, dstPoint);
		sourcecam->modelMat = transform * sourcecam->modelMat;
		sourcecam->calibrated = true;
	}
};

typedef struct calibrateResult {
	std::vector<glm::vec3> points;
	glm::mat4 calibMat;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	bool success;
}CalibrateResult;

class VirtualCam {
public :
	int w, h;
	float ppx, ppy, fx, fy;
	
	glm::vec3 position;
	glm::vec3 rotate;

	uint16_t* p_depth_frame;
	uchar* p_depth_color_frame;
	uchar* p_color_frame;

	GLuint image, depthImage;
	//helper 
	GLuint camIconVao, camIconVbo;
	
	unsigned int framebuffer;
	unsigned int texColorBuffer;
	unsigned int rbo;

	glm::mat4 getModelMat() {
		
		glm::mat4 identity = glm::mat4(1.0);
		glm::mat4 r = glm::translate(identity, position);
		glm::mat4 rt = glm::rotate(r, rotate.y, glm::vec3(0, 1.0, 0.0));
		return rt;
	}

	VirtualCam(int width,int height) {
		w = width;
		h = height;
		glGenTextures(1, &image);
		glGenTextures(1, &depthImage);
		glGenVertexArrays(1, &camIconVao);
		glGenBuffers(1, &camIconVbo);
		p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
		p_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
		p_depth_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
		position = glm::vec3(0.452, 0, 0.186);
		rotate = glm::vec3(0, 5.448, 0);

		glGenFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glGenTextures(1, &texColorBuffer);
		glBindTexture(GL_TEXTURE_2D, texColorBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		// attach it to currently bound framebuffer object
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);

		glGenRenderbuffers(1, &rbo);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	~VirtualCam() {
		glDeleteVertexArrays(1, &camIconVao);
		glDeleteBuffers(1, &camIconVbo);
		glDeleteTextures(1, &image);
		glDeleteTextures(1, &depthImage);
		free((void*)p_depth_frame);
		free((void*)p_color_frame);
		free((void*)p_depth_color_frame);
	}
};

class RealsenseGL {

public :
	bool use2createMesh = true;
	bool ready2Delete = false;
	RealsenseDevice* camera;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	GLuint vao, vbo, image, depthImage;

	//helper 
	GLuint camIconVao, camIconVbo;

	RealsenseGL() {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenTextures(1, &image);
		glGenTextures(1, &depthImage);
		glGenVertexArrays(1, &camIconVao);
		glGenBuffers(1, &camIconVbo);
	}
	~RealsenseGL() {
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &camIconVao);
		glDeleteBuffers(1, &camIconVbo);
		glDeleteTextures(1, &image);
		glDeleteTextures(1, &depthImage);
	}
};

class PointcloudApp :public ImguiOpeGL3App {

	//cuda opengl
	int* cudaIndicesCount = 0;
	uint16_t* cudaDepthData = 0;
	uchar* cudaColorData = 0;
	GLuint vao, vbo, ibo;
	struct cudaGraphicsResource *cuda_vbo_resource, *cuda_ibo_resource;
	int width = 1280;
	int height = 720;

	GLuint shader_program,texture_shader_program,project_shader_program;
	GLuint axisVao, axisVbo;
	
	// mesh reconstruct 
	GLuint meshVao, meshVbo, meshibo;
	GLfloat* vertices = nullptr;
	int verticesCount = 0;
	unsigned int *indices = nullptr;
	int indicesCount = 0;
	bool wireframe = false;
	bool renderMesh = false;
	long time = 0;
	float searchRadius = 0.1;
	int maximumNearestNeighbors = 30;
	float maximumSurfaceAngle = M_PI / 4;

	int pointcloudDensity = 1;

	std::vector<VirtualCam*> virtualcams;

	rs2::context ctx;
	std::vector<RealsenseGL> realsenses;
	std::set<std::string> serials;

	float t,pointsize=0.1f;

	CorrespondPointCollector* calibrator=nullptr;
	int collectPointCout = 15;
	float collectthreshold = 0.1f;

public:
	PointcloudApp():ImguiOpeGL3App(){
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
	}
	~PointcloudApp() {
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			removeDevice(device);
		}
		glDeleteVertexArrays(1, &axisVao);
		glDeleteBuffers(1, &axisVbo);

		glDeleteVertexArrays(1, &meshVao);
		glDeleteBuffers(1, &meshVbo);
		glDeleteBuffers(1, &meshibo);

		glDeleteVertexArrays(1, &vao);
		CudaOpenGL::deleteVBO(&vbo, cuda_vbo_resource);
		CudaOpenGL::deleteVBO(&ibo, cuda_ibo_resource);
	}

	void addGui() override {
		{
			ImGui::Begin("Virtual cameras: ");
			if (ImGui::Button("Add virtual camera")) {
				if (realsenses.size()) {
					int w = realsenses.begin()->camera->width;
					int h = realsenses.begin()->camera->height;
					VirtualCam* v = new VirtualCam(w, h);
					v->fx = realsenses.begin()->camera->intri.fx;
					v->fy = realsenses.begin()->camera->intri.fy;
					v->ppx = realsenses.begin()->camera->intri.ppx;
					v->ppy = realsenses.begin()->camera->intri.ppy;
					virtualcams.push_back(v);
				}
			}
			for (int i = 0; i < virtualcams.size();i++) {
				ImGui::Text((std::string("Virtual cam") + std::to_string(i)).c_str());
				ImGui::SliderFloat((std::string("PosX##virtualPos") + std::to_string(i)).c_str(), &virtualcams[i]->position.x, -5.0, 5.0f);
				ImGui::SliderFloat((std::string("PosY##virtualPos") + std::to_string(i)).c_str(), &virtualcams[i]->position.y, -5.0, 5.0f);
				ImGui::SliderFloat((std::string("PosZ##virtualPos") + std::to_string(i)).c_str(), &virtualcams[i]->position.z, -5.0, 5.0f);
				ImGui::SliderFloat((std::string("RotateY##virtualRot") + std::to_string(i)).c_str(), &virtualcams[i]->rotate.y, 0, M_PI*2);
			}
			ImGui::End();
		}
		/*{
			ImGui::Begin("Reconstruct: ");
			ImGui::Text("Reconstruct Time : %d",time);
			ImGui::Text("Vetices : %d",verticesCount);
			ImGui::Text("Faces : %d",indicesCount/3);

			ImGui::SliderInt("PointcloudDecimate", &pointcloudDensity, 1,10);
			ImGui::SliderFloat("Radius", &searchRadius, 0,1);
			ImGui::SliderInt("maxNeighbor", &maximumNearestNeighbors, 5,100);
			ImGui::SliderFloat("maxSurfaceAngle", &maximumSurfaceAngle, 0,M_PI*2);

			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
				ImGui::Checkbox((device->camera->serial+ std::string("##recon")).c_str(), &(device->use2createMesh));
			}
			if (ImGui::Button("Reconstruct")) {
				clock_t start = clock();
				verticesCount = 0;
				for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
					if (device->use2createMesh) {
						verticesCount += device->camera->vaildVeticesCount;
					}
				}

				if (vertices != nullptr) {
					free(vertices);
				}
				vertices = (float*)calloc(verticesCount * 6, sizeof(float));

				int currentEmpty = 0;
				for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
					if (device->use2createMesh) {
						memcpy(vertices + currentEmpty,
							device->camera->vertexData, 
							device->camera->vaildVeticesCount * 6 * sizeof(float));
						currentEmpty += device->camera->vaildVeticesCount * 6;
					}
				}

				if (indices != nullptr) {
					free(indices);
				}
				indices = fast_triangulation_of_unordered_pcd(
					vertices,
					verticesCount,
					indicesCount,
					searchRadius,
					maximumNearestNeighbors,
					maximumSurfaceAngle
				);

				clock_t end_t = clock();
				time = end_t - start;
				renderMesh = true;
			}
			ImGui::Checkbox("wireframe", &wireframe);
			ImGui::Checkbox("renderMesh", &renderMesh);

			ImGui::End();
		}*/
		{
			ImGui::Begin("Aruco calibrate: ");
			ImGui::SliderFloat("Threshold", &collectthreshold, 0.05f, 0.3f);
			ImGui::SliderInt("ExpectCollect Point Count", &collectPointCout, 3, 50);
			if (calibrator != nullptr) {
				if (ImGui::Button("cancel calibrate")) {
					delete calibrator;
					calibrator = nullptr;
				}
			}
			ImGui::End();
		}

		{
			ImGui::Begin("Realsense pointcloud viewer: ");                          // Create a window called "Hello, world!" and append into it.
			
			ImGui::SliderFloat("Point size", &pointsize, 0.5f, 50.0f);

			// input url for network device
			static char url[20] = "192.168.0.106";
			ImGui::Text("Network device Ip: ");
			ImGui::SameLine();
			if (ImGui::Button("add")) {
				addNetworkDevice(url);
			}
			ImGui::SameLine();
			ImGui::InputText("##urlInput", url, 20);

			// list all usb3.0 realsense device
			if (ImGui::Button("Refresh"))
				serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
			ImGui::SameLine();
			ImGui::Text("Realsense device :");

			// waiting active device
			for (std::string serial : serials) {
				bool alreadyStart = false;
				for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
					if (device->camera->serial == serial) {
						alreadyStart = true;
						break;
					}
				}
				if (!alreadyStart)
				{
					if (ImGui::Button(serial.c_str())) {
						addDevice(serial);
					}
				}
			}

			// Running device
			ImGui::Text("Running Realsense device :");
			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {				
				
				if (ImGui::Button((std::string("stop##") + device->camera->serial).c_str())) {
					device->ready2Delete = true;
				}
				ImGui::SameLine();
				ImGui::Text(device->camera->serial.c_str());
				ImGui::SameLine();
				ImGui::Checkbox((std::string("visible##") + device->camera->serial).c_str(), &(device->camera->visible));
				ImGui::SameLine();
				ImGui::Checkbox((std::string("calibrated##") + device->camera->serial).c_str(), &(device->camera->calibrated));

				ImGui::SliderFloat((std::string("clip-z##") + device->camera->serial).c_str(), &device->camera->farPlane, 0.5f, 15.0f);
				//ImGui::SameLine();
				//ImGui::Checkbox((std::string("OpencvWindow##") + device->camera->serial).c_str(), &(device->camera->opencvImshow));
			}
			ImGui::End();
		}		
	}

	void removeDevice(std::vector<RealsenseGL>::iterator& device) {
		delete device->camera;		
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
		device = realsenses.erase(device);
	}

	void addNetworkDevice(std::string url) try {
		RealsenseGL device;

		device.camera = new RealsenseDevice();		
		std::string serial= device.camera->runNetworkDevice(url, ctx);

		realsenses.push_back(device);
	}
	catch (const rs2::error& e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	}

	void addDevice(std::string serial) {
		RealsenseGL device;

		device.camera = new RealsenseDevice();
		device.camera->runDevice(serial.c_str(), ctx);

		glGenVertexArrays(1, &device.vao);
		glGenBuffers(1, &device.vbo);

		realsenses.push_back(device);
	}

	void initGL() override {
		shader_program = ImguiOpeGL3App::genPointcloudShader(this->window);
		texture_shader_program = ImguiOpeGL3App::genTextureShader(this->window);
		project_shader_program = ImguiOpeGL3App::genprojectShader(this->window);
		glGenVertexArrays(1, &axisVao);
		glGenBuffers(1, &axisVbo);

		glGenVertexArrays(1, &meshVao);
		glGenBuffers(1, &meshVbo);
		glGenBuffers(1, &meshibo);

		glGenVertexArrays(1, &vao);
		CudaOpenGL::createBufferObject(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsNone, width * height * 6 * sizeof(float), GL_ARRAY_BUFFER);
		CudaOpenGL::createBufferObject(&ibo, &cuda_ibo_resource, cudaGraphicsMapFlagsNone, width * height * 2 * 3 * sizeof(sizeof(unsigned int)), GL_ELEMENT_ARRAY_BUFFER);
		cudaMalloc((void**)&cudaIndicesCount, sizeof(int));
		cudaMalloc((void**)&cudaDepthData, width * height * sizeof(uint16_t));
		cudaMalloc((void**)&cudaColorData, width * height * 3 *sizeof(uchar));
	}

	void collectCalibratePoints() {

		std::vector<glm::vec2> cornerSrc = OpenCVUtils::opencv_detect_aruco_from_RealsenseRaw(
			calibrator->sourcecam->width,
			calibrator->sourcecam->height,
			calibrator->sourcecam->p_color_frame
		);

		std::vector<glm::vec2> cornerTrg = OpenCVUtils::opencv_detect_aruco_from_RealsenseRaw(
			calibrator->targetcam->width,
			calibrator->targetcam->height,
			calibrator->targetcam->p_color_frame
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
					if(point.z==0)	return;
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

		//calibrator->sourcecam->calibrated = true;
	}

	void alignDevice2calibratedDevice(RealsenseDevice* uncalibratedCam) {

		RealsenseDevice* baseCamera = nullptr;
		glm::mat4 baseCam2Markerorigion;
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
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
				uncalibratedCam->modelMat = baseCamera->modelMat*glm::inverse(baseCam2Markerorigion)* c.calibMat;
				
				calibrator = new CorrespondPointCollector(uncalibratedCam,baseCamera,collectPointCout, collectthreshold);
			}
		}
	}

	CalibrateResult putAruco2Origion(RealsenseDevice* camera) {

		CalibrateResult result;
		result.success = false;

		// detect aruco and put tag in origion
		std::vector<glm::vec2> corner = OpenCVUtils::opencv_detect_aruco_from_RealsenseRaw(
			camera->width,
			camera->height,
			camera->p_color_frame
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
				y.x, y.y, y.z, 0.0,
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

	void renderVirtualCameras() {
		glm::mat4 mvp = Projection * View * Model;

		// render virtual camera		
		for (auto virtualcam : virtualcams) {
			glm::mat4 devicemvp = mvp * virtualcam->getModelMat();

			// project pointcloud to image
			memset(virtualcam->p_color_frame, 0, 3 * virtualcam->w * virtualcam->h * sizeof(uchar));
			memset(virtualcam->p_depth_color_frame, 0, 3 * virtualcam->w * virtualcam->h * sizeof(uchar));
			memset(virtualcam->p_depth_frame, UINT16_MAX, virtualcam->w * virtualcam->h * sizeof(uint16_t));
			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
				for (int i = 0; i < device->camera->vaildVeticesCount; i++) {
					glm::vec4 worldPos = glm::vec4(
						device->camera->vertexData[i * 6 + 0],
						device->camera->vertexData[i * 6 + 1],
						device->camera->vertexData[i * 6 + 2],
						1.0
					);

					glm::vec4 uv = glm::inverse(virtualcam->getModelMat()) * worldPos;
					uv.x = uv.x / uv.z * virtualcam->fx + virtualcam->ppx;
					uv.y = uv.y / uv.z * virtualcam->fy + virtualcam->ppy;
					
					int w = int(uv.x);
					int h = int(uv.y);
					if (w>0 && h>0 && w < virtualcam->w && h < virtualcam->h) {
						int index = h * virtualcam->w + w;
						// depth test

						if (virtualcam->p_depth_frame[index] > uv.z) {
							virtualcam->p_depth_frame[index] = uv.z;

							virtualcam->p_color_frame[index * 3 + 2] = device->camera->vertexData[i * 6 + 3] * 255;
							virtualcam->p_color_frame[index * 3 + 1] = device->camera->vertexData[i * 6 + 4] * 255;
							virtualcam->p_color_frame[index * 3 + 0] = device->camera->vertexData[i * 6 + 5] * 255;

							uchar R = 0;
							uchar G = 0;
							uchar B = 0;
							RealsenseDevice::HSVtoRGB((uv.z / 5.0) * 360, 100, 100, R, G, B);
							virtualcam->p_depth_color_frame[index * 3 + 0] = R;
							virtualcam->p_depth_color_frame[index * 3 + 1] = G;
							virtualcam->p_depth_color_frame[index * 3 + 2] = B;
						}
					}
				}
			}

			ImguiOpeGL3App::setTexture(virtualcam->image, virtualcam->p_color_frame, virtualcam->w, virtualcam->h);
			ImguiOpeGL3App::setTexture(virtualcam->depthImage, virtualcam->p_depth_color_frame, virtualcam->w, virtualcam->h);

			// render camera frustum
			ImguiOpeGL3App::genCameraHelper(
				virtualcam->camIconVao, virtualcam->camIconVbo,
				virtualcam->w, virtualcam->h,
				virtualcam->ppx, virtualcam->ppy, virtualcam->fx, virtualcam->fy,
				glm::vec3(1.0, 1.0, 0), 0.2, false
			);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			ImguiOpeGL3App::render(devicemvp, pointsize, shader_program, virtualcam->camIconVao, 3 * 4, GL_TRIANGLES);

			ImguiOpeGL3App::genCameraHelper(
				virtualcam->camIconVao, virtualcam->camIconVbo,
				virtualcam->w, virtualcam->h,
				virtualcam->ppx, virtualcam->ppy, virtualcam->fx, virtualcam->fy,
				glm::vec3(1.0, 1.0, 0), 0.2, true
			);

			std::string uniformNames[] = { "color" ,"depth" };
			GLuint textureId[] = { virtualcam->texColorBuffer,virtualcam->depthImage };
			ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, textureId, 2);
			ImguiOpeGL3App::render(devicemvp, pointsize, texture_shader_program, virtualcam->camIconVao, 3 * 4, GL_TRIANGLES);
		}		
	}

	void renderRealsenses() {
		glm::mat4 mvp = Projection * View * Model;
		// render realsense		
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			glm::mat4 deviceMVP = mvp * device->camera->modelMat;

			if (device->ready2Delete) {
				removeDevice(device);
				device--;
				continue;
			}
			else if (!device->camera->visible) {
				continue;
			}

			device->camera->fetchframes(pointcloudDensity);

			glm::vec3 camhelper = glm::vec3(1, 1, 1);
			bool isCalibratingCamer = false;
			if (calibrator != nullptr) {
				if (device->camera->serial == calibrator->sourcecam->serial) {
					camhelper = glm::vec3(1, 0, 0);
					isCalibratingCamer = true;
					calibrator->render(mvp, shader_program);
				}
				if (device->camera->serial == calibrator->targetcam->serial) {
					camhelper = glm::vec3(0, 1, 0);
					isCalibratingCamer = true;
					calibrator->render(mvp, shader_program);
				}
			}
			else {
				if (device->camera->calibrated) {
					camhelper = glm::vec3(0, 0, 1);
				}
			}

			// render pointcloud
			ImguiOpeGL3App::setPointsVAO(device->vao, device->vbo, device->camera->vertexData, device->camera->vertexCount);
			ImguiOpeGL3App::setTexture(device->image, device->camera->p_color_frame, device->camera->width, device->camera->height);
			ImguiOpeGL3App::setTexture(device->depthImage, device->camera->p_depth_color_frame, device->camera->width, device->camera->height);
			ImguiOpeGL3App::render(mvp, pointsize, shader_program, device->vao, device->camera->vaildVeticesCount, GL_POINTS);

			// render camera frustum
			ImguiOpeGL3App::genCameraHelper(
				device->camIconVao, device->camIconVbo,
				device->camera->width, device->camera->height,
				device->camera->intri.ppx, device->camera->intri.ppy, device->camera->intri.fx, device->camera->intri.fy,
				camhelper, 0.2, false
			);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			ImguiOpeGL3App::render(deviceMVP, pointsize, shader_program, device->camIconVao, 3 * 4, GL_TRIANGLES);

			ImguiOpeGL3App::genCameraHelper(
				device->camIconVao, device->camIconVbo,
				device->camera->width, device->camera->height,
				device->camera->intri.ppx, device->camera->intri.ppy, device->camera->intri.fx, device->camera->intri.fy,
				camhelper, 0.2, true
			);

			std::string uniformNames[] = { "color" ,"depth" };
			GLuint textureId[] = { device->image,device->depthImage };
			ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, textureId, 2);
			ImguiOpeGL3App::render(deviceMVP, pointsize, texture_shader_program, device->camIconVao, 3 * 4, GL_TRIANGLES);

			if (device == realsenses.begin()) {
				device->camera->calibrated = true;
				device->camera->modelMat = glm::mat4(1.0);
			}
			else if (!device->camera->calibrated) {

				if (calibrator == nullptr) {
					alignDevice2calibratedDevice(device->camera);
				}
				else {
					collectCalibratePoints();
				}
			}
		}		
	}
	void renderRealsenseCudaMesh(glm::mat4& mvp,GLuint& program) {
		glBindVertexArray(vao);
		// generate and bind the buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// set up generic attrib pointers
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 0 * sizeof(GLfloat));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 3 * sizeof(GLfloat));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		int count = 0;
		cudaMemcpy(&count, cudaIndicesCount, sizeof(int), cudaMemcpyDeviceToHost);
		ImguiOpeGL3App::renderElements(mvp, pointsize, program, vao, count * 3, GL_FILL);
	}
	void renderRealsenseFrustum() {
		glm::mat4 mvp = Projection * View * Model;

		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			glm::mat4 deviceMVP = mvp * device->camera->modelMat;

			if (device->ready2Delete) {
				removeDevice(device);
				device--;
				continue;
			}
			else if (!device->camera->visible) {
				continue;
			}

			ImguiOpeGL3App::setTexture(device->image, device->camera->p_color_frame, device->camera->width, device->camera->height);
			glm::vec3 camhelper = glm::vec3(1, 1, 1);
			bool isCalibratingCamer = false;
			if (calibrator != nullptr) {
				if (device->camera->serial == calibrator->sourcecam->serial) {
					camhelper = glm::vec3(1, 0, 0);
					isCalibratingCamer = true;
					calibrator->render(mvp, shader_program);
				}
				if (device->camera->serial == calibrator->targetcam->serial) {
					camhelper = glm::vec3(0, 1, 0);
					isCalibratingCamer = true;
					calibrator->render(mvp, shader_program);
				}
			}
			else {
				if (device->camera->calibrated) {
					camhelper = glm::vec3(0, 0, 1);
				}
			}
			// render camera frustum
			ImguiOpeGL3App::genCameraHelper(
				device->camIconVao, device->camIconVbo,
				device->camera->width, device->camera->height,
				device->camera->intri.ppx, device->camera->intri.ppy, device->camera->intri.fx, device->camera->intri.fy,
				camhelper, 0.2, false
			);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			ImguiOpeGL3App::render(deviceMVP, pointsize, shader_program, device->camIconVao, 3 * 4, GL_TRIANGLES);

			ImguiOpeGL3App::genCameraHelper(
				device->camIconVao, device->camIconVbo,
				device->camera->width, device->camera->height,
				device->camera->intri.ppx, device->camera->intri.ppy, device->camera->intri.fx, device->camera->intri.fy,
				camhelper, 0.2, true
			);

			std::string uniformNames[] = { "color" ,"depth" };
			GLuint textureId[] = { device->image,device->depthImage };
			ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, textureId, 2);
			ImguiOpeGL3App::render(deviceMVP, pointsize, texture_shader_program, device->camIconVao, 3 * 4, GL_TRIANGLES);
		}
	}
	void updateRealsenseCuda() {
		glm::mat4 mvp = Projection * View * Model;
		// render realsense		
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			glm::mat4 deviceMVP = mvp * device->camera->modelMat;

			if (device->ready2Delete) {
				removeDevice(device);
				device--;
				continue;
			}
			else if (!device->camera->visible) {
				continue;
			}

			auto copyHost2Device = [this](const void* depthRaw, size_t depthSize, const void* colorRaw, size_t colorSize) {
				cudaMemcpy(cudaDepthData, depthRaw, depthSize, cudaMemcpyHostToDevice);
				cudaMemcpy(cudaColorData, colorRaw, colorSize, cudaMemcpyHostToDevice);
			};
			device->camera->fetchframes(copyHost2Device);

			CudaAlogrithm::depthMap2point(
				&cuda_vbo_resource, 
				cudaDepthData, cudaColorData,
				width, height,
				device->camera->intri.fx, device->camera->intri.fy, device->camera->intri.ppx, device->camera->intri.ppy,
				device->camera->intri.depth_scale,device->camera->farPlane);
			
			CudaAlogrithm::depthMapTriangulate(&cuda_vbo_resource, &cuda_ibo_resource, width, height, cudaIndicesCount);

			renderRealsenseCudaMesh(mvp,shader_program);
		}
	}
	void framebufferRender() override {
		updateRealsenseCuda();

		for (auto virtualcam : virtualcams) {
			glBindFramebuffer(GL_FRAMEBUFFER, virtualcam->framebuffer);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
			glViewport(0, 0, virtualcam->w, virtualcam->h);
			glEnable(GL_DEPTH_TEST);
			
			glm::mat4 m = glm::inverse(virtualcam->getModelMat());
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
				virtualcam->w,
				virtualcam->h,
				virtualcam->fx,
				virtualcam->fy,
				virtualcam->ppx,
				virtualcam->ppy,
				0,
				5
			};
			ImguiOpeGL3App::setUniformFloats(project_shader_program, uniformNames, values,8);
			renderRealsenseCudaMesh(m, project_shader_program);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	}
	
	void mainloop() override {

		glm::mat4 mvp = Projection * View * Model;

		// render center axis
		ImguiOpeGL3App::genOrigionAxis(axisVao, axisVbo);
		glm::mat4 mvpAxis = Projection * View * glm::translate(glm::mat4(1.0),lookAtPoint) * Model;
		ImguiOpeGL3App::render(mvpAxis, pointsize, shader_program, axisVao, 6, GL_LINES);

		// render reconstructed mesh
		if (renderMesh && indicesCount != 0) {
			ImguiOpeGL3App::setTrianglesVAOIBO(meshVao, meshVao, meshibo, vertices, verticesCount, indices, indicesCount);
			ImguiOpeGL3App::renderElements(mvp, pointsize, shader_program, meshVao, indicesCount,
				wireframe? GL_LINE: GL_FILL);
			return;
		}
		
		renderRealsenseCudaMesh(mvp, shader_program);
		renderRealsenseFrustum();
		renderVirtualCameras();
		//renderRealsenses();
	}
};

int main() {
	PointcloudApp pviewer;
	pviewer.initImguiOpenGL3();
}