R"(
#version 330

in vec2 TexCoord;

uniform sampler2D tex1;
uniform sampler2D tex2;

layout(location = 0) out vec4 FragColor;

void main() {
	FragColor = texture(tex2, TexCoord);
	//FragColor = mix(texture(tex1, TexCoord), texture(tex2, TexCoord), 0.2);
}

)"