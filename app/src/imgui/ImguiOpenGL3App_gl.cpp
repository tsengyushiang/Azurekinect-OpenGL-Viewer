#include "ImguiOpeGL3App.h"

void ImguiOpeGL3App::renderElements(glm::mat4& mvp, float psize, GLuint shader_program, GLuint vao, int size,int type) {

    GLuint MatrixID = glGetUniformLocation(shader_program, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

    // use the shader program
    glUseProgram(shader_program);
    // bind the vao
    glBindVertexArray(vao);
    glPolygonMode(GL_FRONT_AND_BACK,type);
    glDrawElements(GL_TRIANGLES, size, GL_UNSIGNED_INT, nullptr);
}

void ImguiOpeGL3App::activateTextures(GLuint shader_program,std::string* uniformName, GLuint* textureId, int textureCount) {

    glUseProgram(shader_program);

    for (int i = 0; i < textureCount; i++) {
        glUniform1i(glGetUniformLocation(shader_program, (uniformName+i)->c_str()), i);
        glActiveTexture(GL_TEXTURE0+i);
        glBindTexture(GL_TEXTURE_2D, *(textureId+i));
     }
}

void ImguiOpeGL3App::render(glm::mat4& mvp,float psize,GLuint shader_program, GLuint vao, float size, int type) {

    GLuint pointsize = glGetUniformLocation(shader_program, "pointsize");
    glUniform1f(pointsize, psize);
    
    GLuint MatrixID = glGetUniformLocation(shader_program, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

    // use the shader program
    glUseProgram(shader_program);
    // bind the vao
    glBindVertexArray(vao);
    // draw
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(type, 0, size);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void ImguiOpeGL3App::setTexture(GLuint& image,const unsigned char* image_data, int width, int height) {

    glBindTexture(GL_TEXTURE_2D, image);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, image_data);
}

void ImguiOpeGL3App::setTrianglesVAOIBO(
    GLuint& vao, GLuint& vbo, GLuint& ibo, 
    GLfloat* vertexData, int vertexSize, 
    unsigned int* indices, int indicesSize) 
{

    // generate and bind the vao
    glBindVertexArray(vao);

    // generate and bind the buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // fill with data
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertexSize * 6, vertexData, GL_STATIC_DRAW);

    // set up generic attrib pointers
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 0 * sizeof(GLfloat));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 3 * sizeof(GLfloat));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize * sizeof(unsigned int), indices, GL_STATIC_DRAW);
}


void ImguiOpeGL3App::setPointsVAO(GLuint& vao, GLuint& vbo,GLfloat* vertexData,float size) {
    
    // input vao vbo need to gen first
    //glGenVertexArrays(1, &vao);
    //glGenBuffers(1, &vbo);    

    // generate and bind the vao
    glBindVertexArray(vao);

    // generate and bind the buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // fill with data
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * size * 6, vertexData, GL_STATIC_DRAW);

    // set up generic attrib pointers
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 0 * sizeof(GLfloat));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (char*)0 + 3 * sizeof(GLfloat));
}

GLuint ImguiOpeGL3App::genTextureShader(GLFWwindow* window) {
    // shader source code
    std::string vertex_source =
#include "shaders/texture.vs"
        ;

    std::string fragment_source =
#include "shaders/texture.fs"
        ;

    // program and shader handles
    GLuint shader_program, vertex_shader, fragment_shader;

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
        return 1;
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
        return 1;
    }

    //glDetachShader(shader_program, vertex_shader);
    //glDetachShader(shader_program, fragment_shader);
    //glDeleteShader(vertex_shader);
    //glDeleteShader(fragment_shader);
    //glDeleteProgram(shader_program);

    return shader_program;
}

GLuint ImguiOpeGL3App::genPointcloudShader(GLFWwindow* window) {
    // shader source code
    std::string vertex_source =
#include "shaders/vertexcolor.vs"
        ;

    std::string fragment_source =
#include "shaders/vertexcolor.fs"
        ;

    // program and shader handles
    GLuint shader_program, vertex_shader, fragment_shader;

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
        return 1;
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
        return 1;
    }

    //glDetachShader(shader_program, vertex_shader);
    //glDetachShader(shader_program, fragment_shader);
    //glDeleteShader(vertex_shader);
    //glDeleteShader(fragment_shader);
    //glDeleteProgram(shader_program);

    return shader_program;
}

// helper to check and display for shader compiler errors
bool ImguiOpeGL3App::check_shader_compile_status(GLuint obj) {
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
bool ImguiOpeGL3App::check_program_link_status(GLuint obj) {
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
