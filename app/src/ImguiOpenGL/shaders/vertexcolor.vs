R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vUv;
layout(location = 2) in vec4 vnormal;

uniform mat4 MVP;
uniform float pointsize;
uniform mat4 modelMat;

out vec4 fcolor;
out vec4 pos;
out vec3 local_pos;
out vec3 normal;
out vec2 TexCoord;

void main() {
    local_pos = vposition.xyz;
    TexCoord = vUv.xy;
    normal = vnormal.xyz;
    
    gl_PointSize = pointsize;
    fcolor = vUv;
    pos = vposition;
    gl_Position =  MVP * vposition;
};

)"