
#include "src/imgui/ImguiOpeGL3App.h"
#include "src/realsnese//RealsenseDevice.h"
#include "src/opencv/opecv-utils.h"

typedef struct calibrateResult {
	glm::mat4 calibMat;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	bool success;
}CalibrateResult;

class RealsenseGL {

public :
	RealsenseDevice* camera;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	GLuint vao, vbo, image;

	RealsenseGL() {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenTextures(1, &image);
	}
	~RealsenseGL() {
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
		glDeleteTextures(1, &image);
	}
};

class PointcloudApp :public ImguiOpeGL3App {
	GLuint shader_program;
	GLuint axisVao, axisVbo;

	int width = 640;
	int height = 480;

	rs2::context ctx;
	std::vector<RealsenseGL> realsenses;
	std::set<std::string> activeDeviceSerials;
	std::set<std::string> serials;

	float t,pointsize=0.1f;
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
	}

	void addGui() override {
		{
			ImGui::Begin("Realsense Gui: ");                          // Create a window called "Hello, world!" and append into it.
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
				if (activeDeviceSerials.find(serial.c_str()) == activeDeviceSerials.end())
				{
					if (ImGui::Button(serial.c_str())) {
						addDevice(serial);
					}
				}
			}

			// Running device
			ImGui::Text("Running Realsense device :");
			for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
				ImGui::TextColored(ImVec4(device->camera->calibrated ? 255 : 0, 255, 255, 255), device->camera->serial.c_str());
				ImGui::SameLine();
				if (ImGui::Button((std::string("stop##") + device->camera->serial).c_str())) {
					removeDevice(device);
					device--;
				}
				else {
					ImGui::SameLine();
					ImGui::Checkbox((std::string("##visible") + device->camera->serial).c_str(), &(device->camera->visible));
					ImGui::SameLine();
					if (ImGui::Button((std::string("calibrate##") + device->camera->serial).c_str())) {
						device->camera->calibrated = !device->camera->calibrated;
					}
					ImGui::SameLine();
					ImGui::Checkbox((std::string("OpencvWindow##") + device->camera->serial).c_str(), &(device->camera->opencvImshow));
					// show image with imgui
					//ImGui::Image((void*)(intptr_t)device->image, ImVec2(device->camera->width, device->camera->height));
				}
			}
			ImGui::End();
		}

		
	}

	void removeDevice(std::vector<RealsenseGL>::iterator device) {
		activeDeviceSerials.erase(device->camera->serial.c_str());

		delete device->camera;

		realsenses.erase(device);
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
	}

	void addNetworkDevice(std::string url) try {
		RealsenseGL device;

		device.camera = new RealsenseDevice();		
		std::string serial= device.camera->runNetworkDevice(url, ctx);
		activeDeviceSerials.insert(serial);	

		realsenses.push_back(device);
	}
	catch (const rs2::error& e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	}

	void addDevice(std::string serial) {
		activeDeviceSerials.insert(serial.c_str());

		RealsenseGL device;

		device.camera = new RealsenseDevice();
		device.camera->runDevice(serial.c_str(), ctx);

		glGenVertexArrays(1, &device.vao);
		glGenBuffers(1, &device.vbo);

		realsenses.push_back(device);
	}

	void initGL() override {
		shader_program = ImguiOpeGL3App::genPointcloudShader(this->window);
		glGenVertexArrays(1, &axisVao);
		glGenBuffers(1, &axisVbo);
	}
	CalibrateResult putAruco2Origion(std::vector<RealsenseGL>::iterator device) {

		CalibrateResult result;

		// detect aruco and put tag in origion
		std::vector<glm::vec2> corner = OpenCVUtils::opencv_detect_aruco_from_RealsenseRaw(
			device->camera->width,
			device->camera->height,
			device->camera->p_depth_frame,
			device->camera->p_color_frame
		);
		result.success = corner.size() > 0;

		if (result.success) {
			glm::vec3 center(0, 0, 0);
			std::vector<glm::vec3> points;
			for (auto p : corner) {
				glm::vec3 point = device->camera->colorPixel2point(p);
				points.push_back(point);
				center += point;
			}
			center /= 4;
			glm::vec3 x = glm::normalize(points[1] - center);
			glm::vec3 z = glm::normalize(points[2] - center);
			glm::vec3 y = glm::vec3(
				x.y * z.z - x.z * z.y,
				x.z * z.x - x.x * z.z,
				x.x * z.y - x.y * z.x
			);
			glm::mat4 tranform = glm::mat4(
				x.x, x.y, x.z, 0.0,
				y.x, y.y, y.z, 0.0,
				z.x, z.y, z.z, 0.0,
				center.x, center.y, center.z, 1.0
			);
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

	void mainloop() override {
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			if (!device->camera->visible)
				continue;
			device->camera->fetchframes();
			ImguiOpeGL3App::setPointsVAO(device->vao, device->vbo, device->camera->vertexData, device->camera->vertexCount);
			ImguiOpeGL3App::setTexture(device->image, device->camera->p_color_frame, device->camera->width, device->camera->height);
			glm::mat4 mvp = Projection * View * device->camera->modelMat;
			ImguiOpeGL3App::render(mvp, pointsize, shader_program, device->vao, device->camera->vertexCount,GL_POINTS);
			
			CalibrateResult c = putAruco2Origion(device);
			if (c.success) {
				device->camera->modelMat = c.calibMat;
			}
		}

		//// draw xyz-axis
		GLfloat axisData[] = {
			//  X     Y     Z           R     G     B
				0.0f, 0.0f, 0.0f,       0.0f, 1.0f, 0.0f, // vertex 0
				0.0f, 0.1f, 0.0f,       0.0f, 1.0f, 0.0f, // vertex 1
				0.0f, 0.0f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 2
				0.1f, 0.0f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 3
				0.0f, 0.0f, 0.0f,       0.0f, 0.0f, 1.0f, // vertex 4
				0.0f, 0.0f, 0.1f,       0.0f, 0.0f, 1.0f, // vertex 5
		};
		ImguiOpeGL3App::setPointsVAO(axisVao, axisVbo, axisData, 6);
		glm::mat4 mvp = Projection * View;
		ImguiOpeGL3App::render(mvp, pointsize, shader_program, axisVao, 6, GL_LINES);
	}
};

int main() {
	PointcloudApp pviewer;
	pviewer.initImguiOpenGL3();
}