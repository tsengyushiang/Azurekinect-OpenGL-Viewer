R"(
#version 330

in vec2 TexCoord;

uniform sampler2D color;
uniform sampler2D depth;

layout(location = 0) out vec4 FragColor;

void main() {
	//FragColor = mix(texture(color, TexCoord), texture(depth, TexCoord), 0.5);
	FragColor = texture(color, TexCoord);
}

)"