#include "ImguiOpeGL3App.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


void ImguiOpeGL3App::glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
void ImguiOpeGL3App::framebufferRender() {}
void ImguiOpeGL3App::onBeforeRender() {}
void ImguiOpeGL3App::onAfterRender() {}
void ImguiOpeGL3App::mainloop() {
    std::cout << "Inheritance public:mainloop() to render your objects." << std::endl;
}
void ImguiOpeGL3App::initGL() {
    std::cout << "Inheritance public:initGL() to create shader,vbo,vao..." << std::endl;
}
void ImguiOpeGL3App::mousedrag(float dx,float dy) {
    //std::cout << "Inheritance public:mousedrag(float dx,float dy) to interact with window" << std::endl;
    
    if (abs(dx) > abs(dy)) {
        if (dx < 0) {
            AzimuthAngle += sensity;
            if (AzimuthAngle > AzimuthAngleMax) {
                AzimuthAngle = AzimuthAnglemin;
            }
        }
        else if (dx > 0) {
            AzimuthAngle -= sensity;
            if (AzimuthAngle < AzimuthAnglemin) {
                AzimuthAngle = AzimuthAngleMax;
            }
        }
    }
    else {
        if (dy < 0) {
            if ((PolarAngle + sensity) > PolarAngleMax) return;
            PolarAngle += sensity;
        }
        else if (dy > 0) {
            if ((PolarAngle - sensity) < PolarAnglemin) return;
            PolarAngle -= sensity;
        }
    }       
}
void ImguiOpeGL3App::addMenu() {
    ImGui::Text("Inherit addMenu() to add custum ui.");               // Display some text (you can use a format strings too)
}
void ImguiOpeGL3App::addGui() {
    ImGui::Text("Inherit addGui() to add custum ui.");               // Display some text (you can use a format strings too)
}
bool ImguiOpeGL3App::addOpenGLPanelGui() {
    return true;
}

void ImguiOpeGL3App::setcamera(float width, float height) {

    AzimuthAngle += autoRotateSpeed;
    if (AzimuthAngle > AzimuthAngleMax) {
        AzimuthAngle = AzimuthAnglemin;
    }

    Projection = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.01f, 100.0f);
    View = glm::lookAt(
        glm::vec3(
            distance * sin(PolarAngle) * cos(AzimuthAngle) + lookAtPoint.x,
            distance * cos(PolarAngle) + lookAtPoint.y,
            distance * sin(PolarAngle) * sin(AzimuthAngle) + lookAtPoint.z), // Camera is at (4,3,3), in World Space
        lookAtPoint, // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    Model = glm::mat4(1.0);
}

void ImguiOpeGL3App::initImguiOpenGL3(int width, int height) {

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
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    initGL();
    main = new GLFrameBuffer(width, height);

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        onBeforeRender();
        time = glfwGetTime();
        framebufferRender();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCullFace(GL_BACK);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        setcamera(display_w, display_h);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // render OpenGL in GUI
        ImGui::Begin("OpenGL");
        {
            bool focused = ImGui::IsWindowFocused();
            main->render([this,&clear_color]() {
                glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                mainloop();
            }, GL_BACK);

            // Using a Child allow to fill all the space of the window.
            // It also alows customization
            // Get the size of the child (i.e. the whole draw size of the windows).
            ImVec2 wsize = ImGui::GetWindowSize();

            float windowWidth = (float)ImGui::GetWindowWidth();
            float windowHeight = (float)ImGui::GetWindowHeight();
            ImGui::GetWindowDrawList()->AddImage(
                (ImTextureID)main->texColorBuffer,
                ImGui::GetWindowPos(), 
                ImVec2(ImGui::GetWindowPos().x+ windowWidth, ImGui::GetWindowPos().y+ windowHeight),
                ImVec2(0, 1), ImVec2(1, 0)
            );
            if (!addOpenGLPanelGui()) {
                if (focused && ImGui::IsMouseDragging(0)) {
                    ImVec2 mousedelta = ImGui::GetMouseDragDelta();
                    mousedrag(mousedelta.x, mousedelta.y);
                }
            }
        }
        ImGui::End();

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            try {
                static float f = 0.0f;
                static int counter = 0;

                ImGui::Begin("Menu : ");                          // Create a window called "Hello, world!" and append into it.
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

                if (ImGui::CollapsingHeader("OpenGL World")) {

                    ImGui::ColorEdit3("background color", (float*)&clear_color); // Edit 3 floats representing a color

                    ImGui::Text("Time : %d", time);
                    ImGui::Text("Camera parameters : ");
                    ImGui::SliderFloat("fov", &fov, 30.0f, 80.0f);
                    ImGui::SliderFloat("distance", &distance, distancemin, distanceMax);
                    ImGui::SliderFloat("lookAt-X", &lookAtPoint.x, -10.0f, 10.0f);
                    ImGui::SliderFloat("lookAt-Y", &lookAtPoint.y, -10.0f, 10.0f);
                    ImGui::SliderFloat("lookAt-Z", &lookAtPoint.z, -10.0f, 10.0f);

                    ImGui::Text("Mouse dragging : ");
                    ImGui::SliderFloat("autoRotateSpeed", &autoRotateSpeed, 0.0, 1e-1);
                    ImGui::SliderFloat("sensity", &sensity, 1e-1, 1e-3);
                    ImGui::SliderFloat("PolarAngle", &PolarAngle, PolarAnglemin, PolarAngleMax);
                    ImGui::SliderFloat("AzimuthAngle", &AzimuthAngle, AzimuthAnglemin, AzimuthAngleMax);
                }
                addMenu();
                ImGui::End();
                addGui();

            }
            catch (std::exception const& e) {
                std::cout << e.what()<<std::endl;
            }
        }

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
        onAfterRender();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}