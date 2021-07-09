#include "ImguiOpeGL3App.h"

void ImguiOpeGL3App::renderElements(glm::mat4& mvp, float psize, GLuint shader_program, GLuint vao, int size,int type) {

    glUseProgram(shader_program);

    GLuint MatrixID = glGetUniformLocation(shader_program, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

    // use the shader program
    // bind the vao
    glBindVertexArray(vao);
    glPolygonMode(GL_FRONT, type);
    glPolygonMode(GL_BACK,GL_NONE);
    glDrawElements(GL_TRIANGLES, size, GL_UNSIGNED_INT, nullptr);

    glUseProgram(0);
}

void ImguiOpeGL3App::setUniformFloats(GLuint shader_program, std::string* uniformName, float* values, int count) {

    glUseProgram(shader_program);

    for (int i = 0; i < count; i++) {
        GLuint uniform = glGetUniformLocation(shader_program, (uniformName + i)->c_str());
        glUniform1f(uniform, values[i]);
    }

    glUseProgram(0);
}

void ImguiOpeGL3App::activateTextures(GLuint shader_program,std::string* uniformName, GLuint* textureId, int textureCount) {

    glUseProgram(shader_program);

    for (int i = 0; i < textureCount; i++) {
        glUniform1i(glGetUniformLocation(shader_program, (uniformName+i)->c_str()), i);
        glActiveTexture(GL_TEXTURE0+i);
        glBindTexture(GL_TEXTURE_2D, *(textureId+i));
    }

    glUseProgram(0);
}

void ImguiOpeGL3App::render(glm::mat4& mvp,float psize,GLuint shader_program, GLuint vao, float size, int type) {

    glUseProgram(shader_program);

    GLuint pointsize = glGetUniformLocation(shader_program, "pointsize");
    glUniform1f(pointsize, psize);
    
    GLuint MatrixID = glGetUniformLocation(shader_program, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

    // use the shader program
    // bind the vao
    glBindVertexArray(vao);
    // draw
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(type, 0, size);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glUseProgram(0);
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