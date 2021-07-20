R"(
#version 330

in vec2 TexCoord;

uniform sampler2D color;
uniform float debug;

layout(location = 0) out vec4 FragColor;

void main() {
	vec4 c = texture(color, TexCoord); 
	if(c.a>0){
		FragColor = c;
	}else{
		FragColor = vec4(debug>0.5?1.0:0.0,0.0,0.0,debug>0.5?1.0:0.0);
	}
}

)"