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

    //uv.w = 1.0;
    //uv.x = (camCoord.x / camCoord.z * fx + ppx) / w * 2.0 - 1.0;
    //uv.y = (camCoord.y / camCoord.z * fy + ppy) / h * 2.0 - 1.0;
    //uv.z = (camCoord.z/far*2.0-1.0);

    uv.w = uv.z;
    uv.x = (uv.x * (2.0 / w * fx) + (ppx * 2.0 / w - 1.0) * uv.z);
    uv.y = (uv.y * (2.0 / h * fy) + (ppy * 2.0 / h - 1.0) * uv.z);
    uv.z = (uv.z/far*2.0-1.0)*uv.z;

    fcolor = vcolor;
    gl_Position =  uv;
};

)"