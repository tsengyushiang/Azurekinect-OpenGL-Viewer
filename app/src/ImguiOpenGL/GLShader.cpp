#include "ImguiOpeGL3App.h"

std::map<std::string, std::string> GLShader::shaderLibs = std::map<std::string, std::string>();

void GLShader::loadshaders() {

    shaderLibs["cosWeightDiscardwTexture.fs"] =
#include "shaders/cosWeightDiscardwTexture.fs"
        ;

    shaderLibs["mask.fs"] =
#include "shaders/mask.fs"
        ;

    shaderLibs["cosWeight.fs"] =
#include "shaders/cosWeight.fs"
        ;

    shaderLibs["facenormal.fs"] =
#include "shaders/facenormal.fs"
        ;

    shaderLibs["projectOnScreen.vs"] =
#include "shaders/projectOnScreen.vs"
        ;

    shaderLibs["projectPointcloud.vs"] =
#include "shaders/projectPointcloud.vs"
        ;

    shaderLibs["projectPointcloud.fs"] =
#include "shaders/projectPointcloud.fs"
        ;

    shaderLibs["projectTexture.fs"] =
#include "shaders/projectTexture.fs"
        ;

    shaderLibs["renderMeshUvTexture.vs"] =
#include "shaders/renderMeshUvTexture.vs"
        ;

    shaderLibs["texture.vs"] =
#include "shaders/texture.vs"
        ;

    shaderLibs["texture.fs"] =
#include "shaders/texture.fs"
        ;

    shaderLibs["vertexcolor.vs"] =
#include "shaders/vertexcolor.vs"
        ;
    shaderLibs["vertexcolor.fs"] =
#include "shaders/vertexcolor.fs"
        ;
}

GLuint GLShader::genShaderProgram(GLFWwindow* window, std::string vertexshader, std::string fragmentshdaer) {
    if (shaderLibs.size()==0) {
        loadshaders();
    }

    // program and shader handles
    GLuint shader_program, vertex_shader, fragment_shader;

    return compileAndLink(shaderLibs[vertexshader], shaderLibs[fragmentshdaer], shader_program, vertex_shader, fragment_shader, window);
}

GLuint GLShader::compileAndLink(
    std::string vertex_source, std::string fragment_source,
    GLuint& shader_program, GLuint& vertex_shader, GLuint& fragment_shader, GLFWwindow* window) {
    // we need these to properly pass the strings
    const char* source;
    int length;

    // create and compiler vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    source = vertex_source.c_str();
    length = vertex_source.size();
    glShaderSource(vertex_shader, 1, &source, &length);
    glCompileShader(vertex_shader);
    if (!check_shader_compile_status(vertex_shader)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // create and compiler fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    source = fragment_source.c_str();
    length = fragment_source.size();
    glShaderSource(fragment_shader, 1, &source, &length);
    glCompileShader(fragment_shader);
    if (!check_shader_compile_status(fragment_shader)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // create program
    shader_program = glCreateProgram();

    // attach shaders
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);

    // link the program and check for errors
    glLinkProgram(shader_program);
    if (!check_program_link_status(shader_program)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    return shader_program;
}

// helper to check and display for shader compiler errors
bool GLShader::check_shader_compile_status(GLuint obj) {
    GLint status;
    glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(length);
        glGetShaderInfoLog(obj, length, &length, &log[0]);
        std::cerr << &log[0];
        return false;
    }
    return true;
}

// helper to check and display for shader linker error
bool GLShader::check_program_link_status(GLuint obj) {
    GLint status;
    glGetProgramiv(obj, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(length);
        glGetProgramInfoLog(obj, length, &length, &log[0]);
        std::cerr << &log[0];
        return false;
    }
    return true;
}
