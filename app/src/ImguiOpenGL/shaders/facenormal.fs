R"(
#version 330

in vec3 world_pos;
in vec3 camworld_pos;
layout(location = 0) out vec4 FragColor;

void main() {
    vec3 x = dFdx(world_pos);
    vec3 y = dFdy(world_pos);
    vec3 normal = cross(x, y);
    vec3 norm = normalize(normal);
    FragColor = vec4(norm*0.5+0.5,1.0);

}

)"