
#include "src/imgui/ImguiOpeGL3App.h"

class PointcloudApp :public ImguiOpeGL3App {

	GLuint shader_program;

	// pointcloud datas {x1,y1,z1,r1,g1,b1,...}
	GLuint vao,vbo;

	float t;
public:
	PointcloudApp():ImguiOpeGL3App(){}
	~PointcloudApp() {
		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &vbo);
	}

	void initGL() override {
		shader_program = ImguiOpeGL3App::genPointcloudShader(this->window);

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

	}
	void mainloop() override {
		t += 0.1;
		// data for a fullscreen quad
		GLfloat vertexData[] = {
			//  X     Y     Z           R     G     B
				0.5f, 0.5f, 0.0f,       0.5f, 0.0f, 0.0f, // vertex 0
				-0.5f, 0.5f, 0.0f,       0.0f, 0.5f, 0.0f, // vertex 1
				0.5f,-0.5f, 0.0f,       0.0f, 0.0f, 0.5f, // vertex 2
				0.5f,-0.5f, 0.0f,       0.0f, 0.0f, 0.5f, // vertex 3
				-0.5f, 0.5f, 0.0f,       0.0f, 0.5f, 0.0f, // vertex 4
				-0.5f,-0.5f, 0.0f,       0.5f, 0.0f, 0.0f, // vertex 5
				cos(t),0.0f, sin(t),       0.5f, 0.5f, 0.5f, // vertex 6
		}; // 6 vertices with 6 components (floats) each
		ImguiOpeGL3App::setPointsVAO(vao, vbo, vertexData, 7);

		ImguiOpeGL3App::renderPoints(mvp, 10.0, shader_program, vao, 7);
		//std::cout << "Render" << std::endl;
	}
	void mousedrag(float dx, float dy) override {}
};

int main() {
	PointcloudApp pviewer;
	pviewer.initImguiOpenGL3();
}