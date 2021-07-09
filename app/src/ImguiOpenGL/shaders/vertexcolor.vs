R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vcolor;

uniform mat4 MVP;
uniform float pointsize;

out vec4 fcolor;
out vec4 pos;

void main() {
    gl_PointSize = pointsize;
    fcolor = vcolor;
    pos = vposition;
    gl_Position =  MVP * vposition;
};

)"