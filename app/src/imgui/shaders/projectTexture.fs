R"(
#version 330

in vec4 fcolor;
in vec4 pos;

layout(location = 0) out vec4 FragColor;

uniform mat4 extrinsic;
uniform float w;
uniform float h;
uniform float fx;
uniform float fy;
uniform float ppx;
uniform float ppy;
uniform float near;
uniform float far;
uniform sampler2D color;
uniform sampler2D depthtest;

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
        extrinsic,
        pos,
        w,
        h,
        fx,
        fy,
        ppx,
        ppy,
        near,
        far
    );
    float depth = texture(depthtest, uv.xy).x * far;
    if((uv.z-depth)>1e-3){
        FragColor = vec4(1.0,0.0,0.0,1.0);
    }else{
        FragColor = texture(color, uv.xy);
    }
}

)"