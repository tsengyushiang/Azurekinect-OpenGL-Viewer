R"(
#version 330
layout(location = 0) out vec4 FragColor;
uniform float weightThreshold;

in vec3 world_pos;
in vec3 camworld_pos;

void main() {
    vec3 x = dFdx(world_pos);
    vec3 y = dFdy(world_pos);
    vec3 normal = cross(x, y);
    vec3 norm = normalize(normal);
    float weight = dot(norm,normalize(world_pos-camworld_pos));
    if(weight>weightThreshold){
        FragColor = vec4(weight,weight,weight,1.0);
    }else{
        FragColor = vec4(1.0,0.0,0.0,1.0);
    }

}

)"