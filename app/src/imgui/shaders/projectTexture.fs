R"(
#version 330

in vec4 fcolor;
in vec4 pos;

#define MAXCAM 5

layout(location = 0) out vec4 FragColor;

uniform mat4 extrinsic[MAXCAM];
uniform float w[MAXCAM];
uniform float h[MAXCAM];
uniform float fx[MAXCAM];
uniform float fy[MAXCAM];
uniform float ppx[MAXCAM];
uniform float ppy[MAXCAM];
uniform float near[MAXCAM];
uniform float far[MAXCAM];
uniform sampler2D color[MAXCAM];
uniform sampler2D depthtest[MAXCAM];
uniform float count;

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
    vec4 accumColor = vec4(0,0,0,1.0);
    int visibleCount = 0 ;

    for(int i=0;i<count;i++){
         vec3 uv = projectUv(
            extrinsic[i],
            pos,
            w[i],
            h[i],
            fx[i],
            fy[i],
            ppx[i],
            ppy[i],
            near[i],
            far[i]
        );
        float depth = texture(depthtest[i], uv.xy).x * far[i];
        if((uv.z-depth)>bias){
            //FragColor = vec4(0.0,0.0,0.0,1.0);
        }else{
            accumColor += texture(color[i], uv.xy);
            visibleCount++;
        }
    }
    if(visibleCount>0){
        FragColor = accumColor/visibleCount;
    }else{
        FragColor = vec4(0.0,0.0,0.0,1.0);
    }
}

)"