R"(
#version 330
layout(location = 0) out vec4 FragColor;
uniform float weightThreshold;

in vec3 world_pos;
in vec3 camworld_pos;
in vec3 normal;

void main() {

    float weight = dot(normal,normalize(camworld_pos-world_pos));
    if(weight>weightThreshold){
        FragColor = vec4(weight,weight,weight,1.0);
    }else{
        FragColor = vec4(1.0,0.0,0.0,1.0);
    }

}

)"