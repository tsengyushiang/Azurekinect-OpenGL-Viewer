R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vcolor;

uniform mat4 MVP;
uniform float pointsize;

out vec4 fcolor;

void main() {
    gl_PointSize = pointsize;
    fcolor = vcolor;
    gl_Position =  MVP * vposition;
};

)"