
#include "src/imgui/ImguiOpeGL3App.h"
#include "src/realsnese//RealsenseDevice.h"

class PointcloudApp :public ImguiOpeGL3App {

	GLuint shader_program;

	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}
	GLuint vao,vbo;
	int vertexCount;
	GLfloat* vertexData;

	RealsenseDevice cam;

	float t,pointsize=0.1f;
public:
	PointcloudApp():ImguiOpeGL3App(){}
	~PointcloudApp() {
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
		free(vertexData);
	}

	void addGui() override {

	}
	void initGL() override {
		shader_program = ImguiOpeGL3App::genPointcloudShader(this->window);

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		cam.runNetworkDevice("192.168.0.106");

		vertexCount = cam.width * cam.height;
		vertexData = (GLfloat*)calloc(6 * vertexCount, sizeof(GLfloat)); // 6 represent xyzrgb

	}
	void mainloop() override {
		if (cam.fetchframes(true)) {
			
			for (int i = 0; i < cam.height; i++) {
				for (int j = 0; j < cam.width; j++) {
					int index = i * cam.width + j;
					if (cam.p_depth_frame) {
						float depthValue = (float)cam.p_depth_frame[index] * cam.intri.depth_scale;
						vertexData[index * 6 + 0] = (float(j) - cam.intri.ppx) / cam.intri.fx * depthValue;
						vertexData[index * 6 + 1] = (float(i) - cam.intri.ppy) / cam.intri.fy * depthValue;
						vertexData[index * 6 + 2] = depthValue;
					}

					if (cam.p_color_frame) {
						vertexData[index * 6 + 3] = (float)cam.p_color_frame[index * 3 + 2] / 255;
						vertexData[index * 6 + 4] = (float)cam.p_color_frame[index * 3 + 1] / 255;
						vertexData[index * 6 + 5] = (float)cam.p_color_frame[index * 3 + 0] / 255;
					}
				}
			}
		}
		ImguiOpeGL3App::setPointsVAO(vao, vbo, vertexData, vertexCount);

		ImguiOpeGL3App::renderPoints(mvp, pointsize, shader_program, vao, vertexCount);
		//std::cout << "Render" << std::endl;
	}
	void mousedrag(float dx, float dy) override {}
};

int main() {
	PointcloudApp pviewer;
	pviewer.initImguiOpenGL3();
}