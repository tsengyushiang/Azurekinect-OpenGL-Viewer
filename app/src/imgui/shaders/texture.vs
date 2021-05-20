R"(
#version 330

layout(location = 0) in vec4 vposition;
layout(location = 1) in vec4 uv;

uniform mat4 MVP;

out vec2 TexCoord;

void main() {
    TexCoord = uv.xy;
    gl_Position =  MVP * vposition;
};

)"