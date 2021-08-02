R"(
#version 330
layout(location = 0) out vec4 FragColor;
uniform float weightThreshold;
uniform sampler2D color;
uniform float debug;

in vec3 local_pos;
in vec2 TexCoord;
in vec3 normal;
void main() {
    float weight = dot(normal,normalize(-local_pos));
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