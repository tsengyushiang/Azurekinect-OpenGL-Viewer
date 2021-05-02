
#include "src/imgui/ImguiOpeGL3App.h"

class App :public ImguiOpeGL3App {

public:
	App():ImguiOpeGL3App(){}

	void initGL() override {
		std::cout <<"create shader, VAO"<<std::endl;
	}
	void mainloop() override {
		std::cout << "Render" << std::endl;
	}
};

int main() {
	App a;
	a.initImguiOpenGL3();
}