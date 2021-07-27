R"(
#version 330

in vec3 world_pos;
in vec3 camworld_pos;
layout(location = 0) out vec4 FragColor;
in vec3 normal;

void main() {
    FragColor = vec4(normal*0.5+0.5,1.0);
}

)"