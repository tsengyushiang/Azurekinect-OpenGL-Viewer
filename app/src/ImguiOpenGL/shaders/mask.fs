R"(
#version 330
layout(location = 0) out vec4 FragColor;
uniform float weightThreshold;
uniform sampler2D color;
uniform float debug;

in vec3 world_pos;
in vec3 camworld_pos;
in vec2 TexCoord;

void main() {
    vec3 x = dFdx(world_pos);
    vec3 y = dFdy(world_pos);
    vec3 normal = cross(x, y);
    vec3 norm = normalize(normal);
    float weight = dot(norm,normalize(world_pos-camworld_pos));
    if(weight>weightThreshold){
        vec4 c = texture(color, TexCoord); 
	    if(c.a>0){
		    FragColor = vec4(1.0,1.0,1.0,1.0);
	    }else{
		    FragColor = vec4(0.0,0.0,0.0,0.0);
	    }
    }
}

)"