
#include "src/imgui/ImguiOpeGL3App.h"
#include "src/realsnese//RealsenseDevice.h"
#include "src/realsnese/JsonRealsenseDevice.h"
#include "src/opencv/opecv-utils.h"
#include "src/pcl/examples-pcl.h"
#include "src/cuda/CudaOpenGLUtils.h"
#include "src/cuda/cudaUtils.cuh"
#include <ctime>
#include "src/json/jsonUtils.h"

#define MAXTEXTURE 5

class CorrespondPointCollector {

public :
	GLuint vao, vbo;

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
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		//sourcecam->opencvImshow = true;
		//targetcam->opencvImshow = true;
	}
	~CorrespondPointCollector() {
		free(source);
		free(target);
		free(result);
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);

		//sourcecam->opencvImshow = false;
		//targetcam->opencvImshow = false;
	}

	void render(glm::mat4 mvp,GLuint shader_program) {
		ImguiOpeGL3App::setPointsVAO(vao, vbo, source, size);
		ImguiOpeGL3App::render(mvp, 10, shader_program, vao, size, GL_POINTS);

		ImguiOpeGL3App::setPointsVAO(vao, vbo, target, size);
		ImguiOpeGL3App::render(mvp, 10, shader_program, vao, size, GL_POINTS);
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

	void calibrate() {
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

	float distance = 1;
	float PolarAngle = 1.57;
	float AzimuthAngle = 4;
	float farplane=1.5;
	float nearplane=0;

	uint16_t* p_depth_frame;
	uchar* p_depth_color_frame;
	uchar* p_color_frame;

	GLuint image, depthImage;
	//helper 
	GLuint camIconVao, camIconVbo;
	
	GLFrameBuffer framebuffer;

	bool visible = true;
	CudaGLDepth2PlaneMesh planemesh;

	uchar* colorRaw;
	float* depthRaw;
	uint16_t* depthintRaw;

	glm::mat4 getModelMat(glm::vec3 lookAtPoint) {

		glm::mat4 rt = glm::lookAt(
			glm::vec3(
				distance * sin(PolarAngle) * cos(AzimuthAngle) + lookAtPoint.x,
				distance * cos(PolarAngle) + lookAtPoint.y,
				distance * sin(PolarAngle) * sin(AzimuthAngle) + lookAtPoint.z), // Camera is at (4,3,3), in World Space
			lookAtPoint, // and looks at the origin
			glm::vec3(0, -1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
		);
		return glm::scale(glm::inverse(rt),glm::vec3(1,-1,-1));
	}

	VirtualCam(int width,int height):planemesh(width, height),framebuffer(width, height){
		w = width;
		h = height;
		colorRaw = new uchar[w * h * 3];
		depthRaw = new float[w * h];
		depthintRaw = new uint16_t[w * h];
		glGenTextures(1, &image);
		glGenTextures(1, &depthImage);
		glGenVertexArrays(1, &camIconVao);
		glGenBuffers(1, &camIconVbo);
		p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
		p_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
		p_depth_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
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
	void save(std::string filename, glm::vec3 lookAtPoint) {
		glm::mat4 modelMat = getModelMat(lookAtPoint);

		updateMeshwithCUDA();

		//for (int i = 0; i < w * h; i++) {
		//	if (dpixels[i] > 0.9999) {
		//		dpixels[i] = 0;
		//	}
		//}

		std::vector<float> extrinsic = {
			modelMat[0][0],modelMat[1][0],modelMat[2][0],modelMat[3][0],
			modelMat[0][1],modelMat[1][1],modelMat[2][1],modelMat[3][1],
			modelMat[0][2],modelMat[1][2],modelMat[2][2],modelMat[3][2],
			modelMat[0][3],modelMat[1][3],modelMat[2][3],modelMat[3][3]
		};

		JsonUtils::saveRealsenseJson(filename,
			w, h,
			fx, fy, ppx, ppy, farplane,
			depthRaw, colorRaw, extrinsic
		);

		//// colorize and visualize
		//cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)pixels, cv::Mat::AUTO_STEP);
		//cv::imshow("virtualview", image);

		//// depthmap colorize
		//GLubyte* dpixelscolor = new GLubyte[w * h * 3];
		//for (int i = 0; i < w * h; i++) {
		//	dpixelscolor[i * 3 + 0] = dpixels[i] * 255;
		//	dpixelscolor[i * 3 + 1] = dpixels[i] * 255;
		//	dpixelscolor[i * 3 + 2] = dpixels[i] * 255;
		//}
		//cv::Mat src(cv::Size(w, h), CV_8UC3, (void*)dpixelscolor, cv::Mat::AUTO_STEP);
		//cv::Mat dst;
		//cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
		//cv::equalizeHist(src, dst);
		//cv::imshow("virtualview-depth", dst);
	}
	// pass realsense data to cuda and compute plane mesh and point cloud
	void updateMeshwithCUDA() {

		framebuffer.getRawData(colorRaw, depthRaw);

		int scale = 10000;
		for (int i = 0; i < w * h; i++) {
			depthintRaw[i] = uint16_t(depthRaw[i] * 10000);
		}

		cudaMemcpy(planemesh.cudaDepthData, depthintRaw, w * h * sizeof(uint16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(planemesh.cudaColorData, colorRaw, 3 * w * h * sizeof(uchar), cudaMemcpyHostToDevice);

		CudaAlogrithm::depthMap2point(
			&planemesh.cuda_vbo_resource,
			planemesh.cudaDepthData, planemesh.cudaColorData,
			planemesh.width, planemesh.height,
			fx, fy, ppx, ppy, farplane/ 10000, farplane);

		CudaAlogrithm::depthMapTriangulate(
			&planemesh.cuda_vbo_resource,
			&planemesh.cuda_ibo_resource,
			planemesh.width,
			planemesh.height,
			planemesh.cudaIndicesCount,
			1
		);
	}

	// render single realsense mesh
	void renderRealsenseCudaMesh(glm::mat4& mvp, GLuint& program) {
		if (!visible)return;

		auto render = [this, mvp, program](GLuint& vao, int& count) {
			glm::mat4 m = mvp;
			ImguiOpeGL3App::renderElements(m, 0, program, vao, count * 3, GL_FILL);
		};

		planemesh.render(render);
	}

};

class RealsenseGL {

public :
	bool render2Virtualview = true;
	bool use2createMesh = true;
	bool ready2Delete = false;
	RealsenseDevice* camera;

	GLuint image, depthImage;

	//helper 
	GLuint camIconVao, camIconVbo;

	CudaGLDepth2PlaneMesh planemesh;

	// project depthbuffer
	GLFrameBuffer framebuffer;

	RealsenseGL():planemesh(1280,720), framebuffer(1280,720){
		camera = new RealsenseDevice();
		glGenTextures(1, &image);
		glGenTextures(1, &depthImage);
		glGenVertexArrays(1, &camIconVao);
		glGenBuffers(1, &camIconVbo);
	}
	void destory() {
		glDeleteVertexArrays(1, &camIconVao);
		glDeleteBuffers(1, &camIconVbo);
		glDeleteTextures(1, &image);
		glDeleteTextures(1, &depthImage);

		planemesh.destory();
	}

	// pass realsense data to cuda and compute plane mesh and point cloud
	void updateMeshwithCUDA(float planeMeshThreshold) {
		auto copyHost2Device = [this](const void* depthRaw, size_t depthSize, const void* colorRaw, size_t colorSize) {
			cudaMemcpy(planemesh.cudaDepthData, depthRaw, depthSize, cudaMemcpyHostToDevice);
			cudaMemcpy(planemesh.cudaColorData, colorRaw, colorSize, cudaMemcpyHostToDevice);
		};
		camera->fetchframes(copyHost2Device);

		CudaAlogrithm::depthMap2point(
			&planemesh.cuda_vbo_resource,
			planemesh.cudaDepthData, planemesh.cudaColorData,
			camera->width, camera->height,
			camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
			camera->intri.depth_scale, camera->farPlane);

		CudaAlogrithm::depthMapTriangulate(
			&planemesh.cuda_vbo_resource, 
			&planemesh.cuda_ibo_resource, 
			camera->width,
			camera->height,
			planemesh.cudaIndicesCount,
			planeMeshThreshold
		);
	}

	// render single realsense mesh
	void renderRealsenseCudaMesh(glm::mat4& mvp, GLuint& program) {
		auto render = [this, mvp, program](GLuint& vao, int& count) {
			glm::mat4 m = mvp * camera->modelMat;
			ImguiOpeGL3App::renderElements(m, 0, program, vao, count * 3, GL_FILL);
		};

		planemesh.render(render);		
	}

	void save() {
		JsonUtils::saveRealsenseJson(
			camera->serial,
			camera->width,camera->height,
			camera->intri.fx, camera->intri.fy, camera->intri.ppx, camera->intri.ppy,
			camera->intri.depth_scale,camera->p_depth_frame,camera->p_color_frame
		);
	}

	void addui() {
		if (ImGui::Button((std::string("stop##") + camera->serial).c_str())) {
			ready2Delete = true;
		}
		ImGui::SameLine();
		ImGui::Text(camera->serial.c_str());
		ImGui::Checkbox((std::string("visible##") + camera->serial).c_str(), &(camera->visible));
		ImGui::SameLine();
		ImGui::Checkbox((std::string("calibrated##") + camera->serial).c_str(), &(camera->calibrated));
		ImGui::Checkbox((std::string("render2virutal##") + camera->serial).c_str(), &render2Virtualview);
		ImGui::SliderFloat((std::string("clip-z##") + camera->serial).c_str(), &camera->farPlane, 0.5f, 15.0f);
		//ImGui::SameLine();
		//ImGui::Checkbox((std::string("OpencvWindow##") + device->camera->serial).c_str(), &(device->camera->opencvImshow));
	}

};

class RealsenseDepthSythesisApp :public ImguiOpeGL3App {

	GLuint shader_program, texture_shader_program;
	GLuint project_shader_program;
	GLuint projectTextre_shader_program, screen_projectTextre_shader_program;
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

	float planeMeshThreshold=1;

	float projectiveTextureIndex = 0;
	float projectDepthBias = 5e-3;
public:
	RealsenseDepthSythesisApp():ImguiOpeGL3App(){
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
	}
	~RealsenseDepthSythesisApp() {
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			removeDevice(device);
		}
		glDeleteVertexArrays(1, &axisVao);
		glDeleteBuffers(1, &axisVbo);

		glDeleteVertexArrays(1, &meshVao);
		glDeleteBuffers(1, &meshVbo);
		glDeleteBuffers(1, &meshibo);
	}

	void addGui() override {
		//virtual camera params
		{
			ImGui::Begin("Virtual cameras: ");
			if (ImGui::Button("save virtual view")) {
				for (int i = 0; i < virtualcams.size(); i++) {
					auto virtualcam = virtualcams[i];
					virtualcam->save(std::string("virtualview") + std::to_string(i),lookAtPoint);
				}
			}
			//if (ImGui::Button("Add virtual camera")) {
			//	//if (realsenses.size()) {
			//	//	int w = realsenses.begin()->camera->width;
			//	//	int h = realsenses.begin()->camera->height;
			//	//	VirtualCam* v = new VirtualCam(w, h);
			//	//	v->fx = realsenses.begin()->camera->intri.fx;
			//	//	v->fy = realsenses.begin()->camera->intri.fy;
			//	//	v->ppx = realsenses.begin()->camera->intri.ppx;
			//	//	v->ppy = realsenses.begin()->camera->intri.ppy;
			//	//	virtualcams.push_back(v);
			//	//}
			//	
			//	VirtualCam* v = new VirtualCam(1280, 720);
			//	v->fx = 924.6023559570313;
			//	v->fy = 922.5956420898438;
			//	v->ppx = 632.439208984375;
			//	v->ppy = 356.8707275390625;
			//	virtualcams.push_back(v);
			//}
			for (int i = 0; i < virtualcams.size();i++) {
				ImGui::Text((std::string("Virtual cam") + std::to_string(i)).c_str());
				ImGui::SliderFloat((std::string("farplane##farplane") + std::to_string(i)).c_str(), &virtualcams[i]->farplane, 0, 10);
				ImGui::SliderFloat((std::string("AzimuthAngle##AzimuthAngle") + std::to_string(i)).c_str(), &virtualcams[i]->AzimuthAngle, AzimuthAnglemin, AzimuthAngleMax);
				ImGui::SliderFloat((std::string("PolarAngle##PolarAngle") + std::to_string(i)).c_str(), &virtualcams[i]->PolarAngle, PolarAnglemin, PolarAngleMax);
				ImGui::SliderFloat((std::string("distance##distance") + std::to_string(i)).c_str(), &virtualcams[i]->distance, distancemin, distanceMax);
				ImGui::Checkbox((std::string("visible##visible") + std::to_string(i)).c_str(), &virtualcams[i]->visible);
			}
			ImGui::End();
		}

		{
			ImGui::Begin("Projective Texture index");			
			ImGui::SliderFloat("projectiveTextureIndex", &projectiveTextureIndex, 0, realsenses.size());
			ImGui::SliderFloat("projectDepthBias", &projectDepthBias, 1e-3, 10e-3);
			ImGui::End();
		}
		//reconstruct params
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
		
		//aruco calibrate feature point collector params
		{
			ImGui::Begin("Calibrate: ");
			if (ImGui::Button("load camera pose")) {
				std::map<std::string, glm::mat4> poses;
				std::vector<Jsonformat::CamPose> setting;
				JsonUtils::loadCameraPoses(setting);
				for (auto cam : setting) {
					poses[cam.id] = glm::mat4(
						cam.extrinsic[0], cam.extrinsic[4], cam.extrinsic[8], cam.extrinsic[12],
						cam.extrinsic[1], cam.extrinsic[5], cam.extrinsic[9], cam.extrinsic[13],
						cam.extrinsic[2], cam.extrinsic[6], cam.extrinsic[10], cam.extrinsic[14],
						cam.extrinsic[3], cam.extrinsic[7], cam.extrinsic[11], cam.extrinsic[15]
					);
				}
				for (auto device : realsenses) {
					device.camera->calibrated = true;
					device.camera->modelMat = poses[device.camera->serial];
				}
			}
			ImGui::SliderFloat("Distance Threshold", &collectthreshold, 0.05f, 0.3f);
			ImGui::SliderInt("ExpectCollect Point Count", &collectPointCout, 3, 50);
			if (calibrator != nullptr) {
				if (ImGui::Button("cancel calibrate")) {
					delete calibrator;
					calibrator = nullptr;
				}
			}
			if (ImGui::Button("save camera pose")) {
				std::vector<Jsonformat::CamPose> setting;
				for (auto device : realsenses) {
					glm::mat4 modelMat = device.camera->modelMat;
					std::vector<float> extrinsic = {
						modelMat[0][0],modelMat[1][0],modelMat[2][0],modelMat[3][0],
						modelMat[0][1],modelMat[1][1],modelMat[2][1],modelMat[3][1],
						modelMat[0][2],modelMat[1][2],modelMat[2][2],modelMat[3][2],
						modelMat[0][3],modelMat[1][3],modelMat[2][3],modelMat[3][3]
					};
					Jsonformat::CamPose c = { device.camera->serial ,extrinsic };
					setting.push_back(c);
				}
				JsonUtils::saveCameraPoses(setting);
			}
			ImGui::End();
		}

		// realsense ui
		{
			ImGui::Begin("Realsense pointcloud viewer: ");                          // Create a window called "Hello, world!" and append into it.
			
			ImGui::SliderFloat("Point size", &pointsize, 0.5f, 50.0f);

			static char jsonfilename[20] = "932122060549";
			ImGui::Text("jsonfilename: ");
			ImGui::SameLine();
			if (ImGui::Button("add##jsonfilename")) {
				addJsonDevice(jsonfilename);
			}
			ImGui::SameLine();
			ImGui::InputText("##jsonfilenameurlInput", jsonfilename, 20);

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
			{
				ImGui::Text("Debug :");
				if (ImGui::Button("snapshot all")) {
					for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
						device->save();
					}
				}
				ImGui::SliderFloat("planeMeshThreshold", &planeMeshThreshold, 1.0f, 90.0f);
			}
			// Running device
			ImGui::Text("Running Realsense device :");
			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {				
				device->addui();
			}
			ImGui::End();
		}		
	}

	void removeDevice(std::vector<RealsenseGL>::iterator& device) {
		delete device->camera;		
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
		device = realsenses.erase(device);
		device->destory();
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

	void addJsonDevice(std::string serial) {
		RealsenseGL device;

		device.camera = new JsonRealsenseDevice();
		JsonUtils::loadRealsenseJson(serial,
			device.camera->width, 
			device.camera->height, 
			device.camera->intri.fx, 
			device.camera->intri.fy, 
			device.camera->intri.ppx, 
			device.camera->intri.ppy, 
			device.camera->intri.depth_scale, 
			&device.camera->p_depth_frame, 
			&device.camera->p_color_frame);
		device.camera->serial = serial;

		realsenses.push_back(device);
	}

	void addDevice(std::string serial) {
		RealsenseGL device;
		device.camera->runDevice(serial.c_str(), ctx);
		realsenses.push_back(device);
	}

	void initGL() override {
		shader_program = ImguiOpeGL3App::genPointcloudShader(this->window);
		texture_shader_program = ImguiOpeGL3App::genTextureShader(this->window);
		project_shader_program = ImguiOpeGL3App::genprojectShader(this->window);
		projectTextre_shader_program = ImguiOpeGL3App::genprojectTextureShader(this->window,true);
		screen_projectTextre_shader_program = ImguiOpeGL3App::genprojectTextureShader(this->window,false);
		glGenVertexArrays(1, &axisVao);
		glGenBuffers(1, &axisVbo);

		glGenVertexArrays(1, &meshVao);
		glGenBuffers(1, &meshVbo);
		glGenBuffers(1, &meshibo);

		VirtualCam* v = new VirtualCam(1280, 720);
		v->fx = 924.6023559570313;
		v->fy = 922.5956420898438;
		v->ppx = 632.439208984375;
		v->ppy = 356.8707275390625;
		virtualcams.push_back(v);
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

	// virtual mesh project depth to real camera (prepared for projective texture)
	void renderPorjectDepth(std::vector<RealsenseGL>::iterator device, glm::vec2 offset) {
		ImguiOpeGL3App::genCameraHelper(
			device->camIconVao, device->camIconVbo,
			1, 1, 0.5, 0.5, 0.5, 0.5, glm::vec3(1, 1, 0), 1.0, true
		);

		std::string uniformNames[] = { "color" };
		GLuint texturedepth[] = { device->framebuffer.depthBuffer };
		ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, texturedepth, 1);
		glm::mat4 screenDepthMVP =
			glm::scale(
				glm::translate(
					glm::mat4(1.0),
					glm::vec3(offset.x, offset.y, 0.0)
				),
				glm::vec3(0.25, -0.25, 1e-3)
			);
		ImguiOpeGL3App::render(screenDepthMVP, pointsize, texture_shader_program, device->camIconVao, 3 * 4, GL_TRIANGLES);
	}

	// virtual view frustum and image(framebuffer)
	void renderVirtualCamera(VirtualCam* virtualcam) {
		glm::mat4 mvp = Projection * View * Model;

		// render virtual camera		
		glm::mat4 devicemvp = mvp * virtualcam->getModelMat(lookAtPoint);

		// render camera frustum
		ImguiOpeGL3App::genCameraHelper(
			virtualcam->camIconVao, virtualcam->camIconVbo,
			virtualcam->w, virtualcam->h,
			virtualcam->ppx, virtualcam->ppy, virtualcam->fx, virtualcam->fy,
			glm::vec3(1.0, 1.0, 0), virtualcam->farplane, false
		);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		ImguiOpeGL3App::render(devicemvp, pointsize, shader_program, virtualcam->camIconVao, 3 * 4, GL_TRIANGLES);

		ImguiOpeGL3App::genCameraHelper(
			virtualcam->camIconVao, virtualcam->camIconVbo,
			1, 1, 0.5, 0.5, 0.5, 0.5, glm::vec3(1,1, 0), 1.0, true
		);

		std::string uniformNames[] = { "color" };
		GLuint texturedepth[] = { virtualcam->framebuffer.depthBuffer};
		ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, texturedepth, 1);
		glm::mat4 screenDepthMVP =
			glm::scale(
				glm::translate(
					glm::mat4(1.0), 
					glm::vec3(-0.75, -0.75, 0.0)
				),
				glm::vec3(0.25, -0.25, 1e-3)
			);
		ImguiOpeGL3App::render(screenDepthMVP, pointsize, texture_shader_program, virtualcam->camIconVao, 3 * 4, GL_TRIANGLES);

		GLuint texturecolor[] = { virtualcam->framebuffer.texColorBuffer};
		ImguiOpeGL3App::activateTextures(texture_shader_program, uniformNames, texturecolor, 1);
		glm::mat4 screencolorMVP =
			glm::scale(
				glm::translate(
					glm::mat4(1.0),
					glm::vec3(-0.25, -0.75, 0.0)
				),
				glm::vec3(0.25, -0.25, 1e-3)
			);		
		ImguiOpeGL3App::render(screencolorMVP, pointsize, texture_shader_program, virtualcam->camIconVao, 3 * 4, GL_TRIANGLES);
		
		rendervirtualMesh(virtualcam,false);
	}

	// render virtual mesh with projective textures
	void rendervirtualMesh(VirtualCam* virtualcam,bool renderOnScreen) {
		glm::mat4 mvp = Projection * View * Model;
		// render virtual camera		
		glm::mat4 devicemvp = mvp * virtualcam->getModelMat(lookAtPoint);

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
			devicemvp = glm::mat4(1.0);
		}

		std::string texNames[MAXTEXTURE];
		GLuint texturs[MAXTEXTURE];

		auto indexTostring = [=](int x) mutable throw() -> std::string
		{
			return std::string("[") + std::to_string(x) + std::string("]");
		};

		for (auto device = realsenses.begin(); device != realsenses.end(); device++)
		{
			texNames[deviceIndex * 2] = "color" + indexTostring(deviceIndex);
			texNames[deviceIndex * 2 + 1] = "depthtest" + indexTostring(deviceIndex);

			texturs[deviceIndex * 2] = device->image;
			texturs[deviceIndex * 2 + 1] = device->framebuffer.depthBuffer;

			glUseProgram(shader);
			GLuint MatrixID = glGetUniformLocation(shader, (std::string("extrinsic") + indexTostring(deviceIndex)).c_str());
			glm::mat4 model = glm::inverse(device->camera->modelMat) * virtualcam->getModelMat(lookAtPoint);
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &model[0][0]);
			glUseProgram(0);

			std::string floatnames[] = {
					"w" + indexTostring(deviceIndex),
					"h" + indexTostring(deviceIndex),
					"fx" + indexTostring(deviceIndex),
					"fy" + indexTostring(deviceIndex),
					"ppx" + indexTostring(deviceIndex),
					"ppy" + indexTostring(deviceIndex),
					"near" + indexTostring(deviceIndex),
					"far" + indexTostring(deviceIndex),
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
			ImguiOpeGL3App::setUniformFloats(shader, floatnames, values, 8);
			deviceIndex++;
		}
		ImguiOpeGL3App::activateTextures(shader, texNames, texturs, deviceIndex * 2);
		std::string indexName[] = { "index","bias" };
		float idx[] = { projectiveTextureIndex,projectDepthBias };
		ImguiOpeGL3App::setUniformFloats(shader, indexName, idx, 2);
		virtualcam->renderRealsenseCudaMesh(devicemvp, shader);
	}
	
	// detect aruco to calibrate unregisted camera
	void waitCalibrateCamera(std::vector<RealsenseGL>::iterator device) {
		if (device == realsenses.begin()) {
			device->camera->calibrated = true;
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
	
	// render single realsense camera pose and color image in world coordinate
	void renderRealsenseFrustum(std::vector<RealsenseGL>::iterator device) {
		glm::mat4 mvp = Projection * View * Model;
		glm::mat4 deviceMVP = mvp * device->camera->modelMat;

		ImguiOpeGL3App::setTexture(device->image, device->camera->p_color_frame, device->camera->width, device->camera->height);
		glm::vec3 camhelper = glm::vec3(1, 1, 1);
		bool isCalibratingCamera = false;
		if (calibrator != nullptr) {
			if (device->camera->serial == calibrator->sourcecam->serial) {
				camhelper = glm::vec3(1, 0, 0);
				isCalibratingCamera = true;
			}
			if (device->camera->serial == calibrator->targetcam->serial) {
				camhelper = glm::vec3(0, 1, 0);
				isCalibratingCamera = true;
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
	
	// render realsense mesh on framebuffer
	void framebufferRender() override {
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			if (device->ready2Delete) {
				removeDevice(device);
				device--;
				continue;
			}
			device->updateMeshwithCUDA(planeMeshThreshold);
		}

		for (auto virtualcam : virtualcams) {
			auto renderForwardWarppedDepth = [virtualcam,this]() {
				glm::mat4 m = glm::inverse(virtualcam->getModelMat(lookAtPoint));
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
					virtualcam->nearplane,
					virtualcam->farplane
				};
				ImguiOpeGL3App::setUniformFloats(project_shader_program, uniformNames, values, 8);
				for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
					if (device->render2Virtualview) {
						device->renderRealsenseCudaMesh(m, project_shader_program);
					}
				}
				virtualcam->updateMeshwithCUDA();
			};
			virtualcam->framebuffer.render(renderForwardWarppedDepth);
		}

		//render project depth
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			auto renderVirtualMeshProjectDepth = [device,this]() {
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

				for (auto virtualcam : virtualcams) {
					glm::mat4 m = glm::inverse(device->camera->modelMat) * virtualcam->getModelMat(lookAtPoint);

					virtualcam->renderRealsenseCudaMesh(m, project_shader_program);
				}
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			};
			device->framebuffer.render(renderVirtualMeshProjectDepth);
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
		
		std::vector<glm::vec2> windows = {
			glm::vec2(0.25,0.75),
			glm::vec2(0.75,0.75),
		};
		int deviceIndex = 0;
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			renderRealsenseFrustum(device);
			renderPorjectDepth(device, windows[deviceIndex++]);
			if (device->camera->visible) {
				device->renderRealsenseCudaMesh(mvp, shader_program);
			}
			waitCalibrateCamera(device);
		}

		for (auto virtualcam : virtualcams) {
			renderVirtualCamera(virtualcam);
		}

		if (calibrator != nullptr) {
			calibrator->render(mvp, shader_program);
		}
	}
};

int main() {
	RealsenseDepthSythesisApp viewer;
	viewer.initImguiOpenGL3();
}