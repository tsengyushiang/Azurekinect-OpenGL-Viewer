
#include "src/imgui/ImguiOpeGL3App.h"
#include "src/realsnese//RealsenseDevice.h"

typedef struct realsenseGL {
	RealsenseDevice* camera;
	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}	
	GLuint vao, vbo;
}RealsenseGL;

class PointcloudApp :public ImguiOpeGL3App {
	GLuint shader_program;

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
	}

	void addGui() override {		

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
		for (auto device = realsenses.begin(); device!=realsenses.end(); device++) {
			if (ImGui::Button((std::string("stop##") + device->camera->serial).c_str())) {
				removeDevice(device);
				device--;
			}
			else {
				ImGui::SameLine();
				ImGui::Checkbox((std::string("OpencvWindow##") + device->camera->serial).c_str(), &(device->camera->opencvImshow));
			}
		}
	}

	void removeDevice(std::vector<RealsenseGL>::iterator device) {
		device->camera->enable = false;

		activeDeviceSerials.erase(device->camera->serial.c_str());

		delete device->camera;
		glDeleteVertexArrays(1, &device->vao);
		glDeleteBuffers(1, &device->vbo);

		realsenses.erase(device);
		serials = RealsenseDevice::getAllDeviceSerialNumber(ctx);
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
	}
	void mainloop() override {
		for (auto device = realsenses.begin(); device != realsenses.end(); device++) {
			device->camera->fetchframes();
			ImguiOpeGL3App::setPointsVAO(device->vao, device->vbo, device->camera->vertexData, device->camera->vertexCount);
			ImguiOpeGL3App::renderPoints(mvp, pointsize, shader_program, device->vao, device->camera->vertexCount);
		}
	}
	void mousedrag(float dx, float dy) override {}
};

int main() {
	PointcloudApp pviewer;
	pviewer.initImguiOpenGL3();
}