R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vcolor;

uniform mat4 MVP;

uniform float w;
uniform float h;
uniform float fx;
uniform float fy;
uniform float ppx;
uniform float ppy;
uniform float near;
uniform float far;

out vec4 fcolor;

void main() {
    vec4 uv = MVP * vposition;
    uv.x = (uv.x / uv.z * fx + ppx) / w * 2.0 - 1.0;
    uv.y = (uv.y / uv.z * fy + ppy) / h * 2.0 - 1.0;
    uv.z /= far * 2.0 - 1.0;

    fcolor = vcolor;
    gl_Position =  uv;
};

)"