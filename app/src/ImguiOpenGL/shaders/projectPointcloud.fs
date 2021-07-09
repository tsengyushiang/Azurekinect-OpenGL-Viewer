R"(
#version 330

in vec4 fcolor;

layout(location = 0) out vec4 FragColor;

uniform float renderIndexColor;
uniform float index_r;
uniform float index_g;
uniform float index_b;

void main() {
	if(renderIndexColor<0.5){
		FragColor = fcolor;
	}else{
		FragColor = vec4(index_r,index_g,index_b,1.0);
	}
}

)"