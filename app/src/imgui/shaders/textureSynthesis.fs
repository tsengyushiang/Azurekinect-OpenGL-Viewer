R"(
#version 330

#define MAXCAM 5

in vec2 TexCoord;

uniform sampler2D color[MAXCAM];
uniform sampler2D indexMap[MAXCAM];
uniform float renderIndexColor;

uniform float count;
uniform float overlapWeights[MAXCAM];

uniform float outliner_r;
uniform float outliner_g;
uniform float outliner_b;

layout(location = 0) out vec4 FragColor;

void main() {
	//outliner
	if(abs(TexCoord.y-0.5)>0.49 || abs(TexCoord.x-0.5)>0.49){
		FragColor = vec4(outliner_r,outliner_g,outliner_b,1.0);
	}else{
        vec4 potentialColor[MAXCAM];
        int visibleIndex[MAXCAM];
        int visibleCount = 0 ;

        for(int i=0;i<count;i++){        
            vec4 c = texture(color[i], TexCoord);
            if(c.a>0.5){
                potentialColor[i] = (renderIndexColor>0.5)? texture(indexMap[i], TexCoord) : c;
                visibleIndex[visibleCount++] = i;
            }
        }

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