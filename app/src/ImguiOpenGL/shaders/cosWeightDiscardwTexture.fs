R"(
#version 330
layout(location = 0) out vec4 FragColor;
uniform float weightThreshold;
uniform sampler2D color;
uniform float debug;

in vec3 world_pos;
in vec3 camworld_pos;
in vec2 TexCoord;
in vec3 normal;

void main() {

    float weight = dot(normal,normalize(camworld_pos-world_pos));
    if(weight>weightThreshold){
        vec4 c = texture(color, TexCoord); 
	    if(c.a>0){
		    FragColor = c;
	    }else{
		    FragColor = vec4(debug>0.5?1.0:0.0,0.0,0.0,debug>0.5?1.0:0.0);
	    }
    }
}

)"