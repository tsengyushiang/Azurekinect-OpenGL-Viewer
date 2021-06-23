R"(
#version 330

in vec4 fcolor;
in vec4 pos;

#define MAXCAM 10

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

uniform float overlapWeights[MAXCAM];
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
    vec4 potentialColor[MAXCAM];
    int visibleIndex[MAXCAM];
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
            
        }else{
            potentialColor[i] = texture(color[i], uv.xy);
            if(potentialColor[i].a>0){
                visibleIndex[visibleCount++] = i;
            }
        }
    }
    if(visibleCount==0){
        // invisible area to all camera
        FragColor = vec4(0.0,0.0,0.0,1.0);
    }else {
        // apply weights to blend texture
        
        vec4 accumColor = vec4(0.0,0.0,0.0,0.0);
        float weightTotal = 0;
        for(int i=0;i<visibleCount;i++){
            int index = visibleIndex[i];
            float weight = overlapWeights[index];

            accumColor += potentialColor[index]*weight;
            weightTotal += weight;
        }

        FragColor = accumColor/weightTotal;
    }
}

)"