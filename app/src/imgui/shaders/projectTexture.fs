R"(
#version 330

in vec4 fcolor;
in vec4 pos;

layout(location = 0) out vec4 FragColor;

uniform mat4 extrinsic[5];
uniform float w[5];
uniform float h[5];
uniform float fx[5];
uniform float fy[5];
uniform float ppx[5];
uniform float ppy[5];
uniform float near[5];
uniform float far[5];
uniform sampler2D color[5];
uniform sampler2D depthtest[5];

uniform float index;
uniform float bias;

vec3 projectUv(
    mat4 inverModelMat,vec4 vposition,
    float w,
    float h,
    float fx,
    float fy,
    float ppx,
    float ppy,
    float near,
    float far
){
	vec4 uv = inverModelMat * vposition;
    uv.x = (uv.x / uv.z * fx + ppx) / w;
    uv.y = (uv.y / uv.z * fy + ppy) / h;
    uv.z = uv.z;
    return uv.xyz;
}

void main() {
    vec3 uv = projectUv(
        extrinsic[int(index)],
        pos,
        w[int(index)],
        h[int(index)],
        fx[int(index)],
        fy[int(index)],
        ppx[int(index)],
        ppy[int(index)],
        near[int(index)],
        far[int(index)]
    );
    float depth = texture(depthtest[int(index)], uv.xy).x * far[int(index)];
    if((uv.z-depth)>bias){
        FragColor = vec4(1.0,0.0,0.0,1.0);
    }else{
        FragColor = texture(color[int(index)], uv.xy);
    }
}

)"