R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vUv;
layout(location = 2) in vec4 vnormal;

uniform mat4 MVP;
uniform mat4 modelMat;

uniform float p_w;
uniform float p_h;
uniform float p_fx;
uniform float p_fy;
uniform float p_ppx;
uniform float p_ppy;
uniform float p_near;
uniform float p_far;

out vec3 world_pos;
out vec3 normal;
out vec3 camworld_pos;
out vec2 TexCoord;

void main() {
    world_pos = (modelMat * vposition).xyz;
    camworld_pos = (modelMat * vec4(0,0,0,1.0)).xyz;
    vec4 uv = MVP * vposition;

    uv.w = uv.z;
    uv.x = (uv.x * (2.0 / p_w * p_fx) + (p_ppx * 2.0 / p_w - 1.0) * uv.z);
    uv.y = (uv.y * (2.0 / p_h * p_fy) + (p_ppy * 2.0 / p_h - 1.0) * uv.z);
    uv.z = (uv.z/p_far*2.0-1.0)*uv.z;

    TexCoord = vUv.xy;
    normal = vnormal.xyz;
    gl_Position =  uv;
};

)"