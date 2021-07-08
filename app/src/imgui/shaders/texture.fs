R"(
#version 330

in vec2 TexCoord;

uniform sampler2D color;
uniform sampler2D depth;

uniform float outliner_r;
uniform float outliner_g;
uniform float outliner_b;
uniform float debug;

layout(location = 0) out vec4 FragColor;

void main() {
	//FragColor = mix(texture(color, TexCoord), texture(depth, TexCoord), 0.5);
	//outliner
	if(abs(TexCoord.y-0.5)>0.49 || abs(TexCoord.x-0.5)>0.49){
		FragColor = vec4(outliner_r,outliner_g,outliner_b,1.0);
	}else{
		vec4 c = texture(color, TexCoord); 
		if(c.a>0){
			FragColor = c;
		}else{
			FragColor = vec4(debug>0.5?1.0:0.0,0.0,0.0,debug>0.5?1.0:0.0);
		}
	}
}

)"