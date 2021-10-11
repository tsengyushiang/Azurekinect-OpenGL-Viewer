R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 vUv;
layout(location = 2) in vec4 vnormal;

uniform mat4 MVP;

out vec2 TexCoord;
out vec3 normal;
out vec3 local_pos;

void main() {
    gl_Position =  vec4(vUv.x*2.0-1.0,vUv.y*2.0-1.0,-1.0,1.0);    
    local_pos = vposition.xyz;
    TexCoord = vUv.xy;
    normal = vnormal.xyz;
};

)"