#include "ImguiOpeGL3App.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


void ImguiOpeGL3App::glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void ImguiOpeGL3App::mainloop() {
    std::cout << "Inheritance public:mainloop() to render your objects." << std::endl;
}
void ImguiOpeGL3App::initGL() {
    std::cout << "Inheritance public:initGL() to create shader,vbo,vao..." << std::endl;
}
void ImguiOpeGL3App::mousedrag(float dx,float dy) {
    std::cout << "Inheritance public:mousedrag(float dx,float dy) to interact with window" << std::endl;
}
void ImguiOpeGL3App::addGui() {
    ImGui::Text("Inherit addGui() to add custum ui.");               // Display some text (you can use a format strings too)
}

void ImguiOpeGL3App::setcamera(float width, float height) {
    glm::mat4 Projection = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.1f, 100.0f);
    glm::mat4 View = glm::lookAt(
        glm::vec3(
            distance * sin(PolarAngle) * cos(AzimuthAngle),
            distance * sin(PolarAngle) * sin(AzimuthAngle),
            distance * cos(PolarAngle)), // Camera is at (4,3,3), in World Space
        glm::vec3(0, 0, 0), // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );

    mvp = Projection * View;
}

void ImguiOpeGL3App::initImguiOpenGL3(float width, float height) {
	
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return ;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    window = glfwCreateWindow(width, height, "Dear ImGui GLFW+OpenGL3 App", NULL, NULL);
    if (window == NULL)
        return ;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (gl3wInit() != 0)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return ;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    initGL();


    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        setcamera(width, height);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_DEPTH_TEST);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);


        mainloop();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (ImGui::IsMouseDragging(0)) {
            ImVec2 mousedelta = ImGui::GetMouseDragDelta();
            mousedrag(mousedelta.x, mousedelta.y);
        }

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Imgui : ");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

            ImGui::Text("Camera parameters.");               // Display some text (you can use a format strings too)
            ImGui::SliderFloat("distance", &distance, 0.0f, 5.0f);            // Edit 1 float using a slider from 0.0f to 0.5f
            ImGui::SliderFloat("PolarAngle", &PolarAngle, -3.14f, 3.14f);            // Edit 1 float using a slider from 0.0f to 0.5f
            ImGui::SliderFloat("AzimuthAngle", &AzimuthAngle, -3.14f, 3.14f);            // Edit 1 float using a slider from 0.0f to 0.5f
            
            addGui();

            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color


            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}