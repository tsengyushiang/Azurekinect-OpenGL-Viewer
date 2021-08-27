#include "InputBase.h"

InputBase::InputBase(int cw, int ch, int dw, int dh)
    :cwidth(cw), cheight(ch), dwidth(dw), dheight(dh)
{
    width = cwidth;
    height = cheight;

    depth_cache = (uint16_t**)calloc(MAXCACHE, sizeof(uint16_t*));
    color_cache = (uchar**)calloc(MAXCACHE, sizeof(uchar*));

    for (int i = 0; i < MAXCACHE; i++) {
        depth_cache[i] = (uint16_t*)calloc(width * height, sizeof(uint16_t));
        color_cache[i] = (uchar*)calloc(INPUT_COLOR_CHANNEL * width * height, sizeof(uchar));
    }

    p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
    p_color_frame = (unsigned char*)calloc(INPUT_COLOR_CHANNEL * width * height, sizeof(unsigned char));

    modelMat = glm::mat4(1.0);
    esitmatePlaneCenter = glm::vec3(0.0,0.0,0.0);
    esitmatePlaneNormal = glm::vec3(0.0,1.0,0.0);
}

InputBase::~InputBase() {
    //free cache mem
    for (int i = 0; i < MAXCACHE; i++) {
        free((void*)depth_cache[i]);
        free((void*)color_cache[i]);
    }
    free((void*)depth_cache);
    free((void*)color_cache);
    
    free((void*)p_depth_frame);
    free((void*)p_color_frame);
}

void InputBase::startRecord(int _recordLength) {
    recordColorFile = fopen(string(serial + "color.bin").c_str(), "wb");
    recordDepthFile = fopen(string(serial + "depth.bin").c_str(), "wb");
    curSaveFrame = 0;
    curRecordFrame = 0;
    maxSave = _recordLength;
    isRecording = true;
    recordThread = new thread(&InputBase::keepTryingSave, this);
}

void InputBase::keepTryingSave() {
    while (isRecording) {
        //check not save too fast
        if (curSaveFrame >= curRecordFrame) {
            continue;
        }

        if (curRecordFrame % 100 == 0) {
            cout << serial << "record" << curRecordFrame << "save" << curSaveFrame << "max" << maxSave << endl;
        }

        if (curRecordFrame - MAXCACHE >= curSaveFrame) {
            cout << serial << "record" << curRecordFrame << "save" << curSaveFrame << "max" << maxSave << endl;
        }

        // save to file
        fwrite(
            color_cache[curSaveFrame%MAXCACHE],
            INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char),
            1,
            recordColorFile
        );
        fflush(recordColorFile);
        fwrite(
            depth_cache[curSaveFrame % MAXCACHE],
            width * height * sizeof(uint16_t),
            1,
            recordDepthFile
        );
        fflush(recordDepthFile);
        
        curSaveFrame++;

        //end record
        if (curSaveFrame >= maxSave) {
            isRecording = false;
            fclose(recordColorFile);
            fclose(recordDepthFile);
        }
    }
}

glm::vec3 InputBase::colorPixel2point(glm::vec2 pixel) {
    int i = pixel.y;
    int j = pixel.x;
    int index = i* width + j;

    float depthValue = (float)p_depth_frame[index] * intri.depth_scale;
    if (depthValue > farPlane) {
        return glm::vec3(0, 0, 0);
    }
    glm::vec3 point(
        (float(j) - intri.ppx) / intri.fx * depthValue,
        (height - 1 - float(i) - intri.ppy) / intri.fy * depthValue,
        depthValue
    );

    return point;
}

bool InputBase::fetchframes(std::function<void(
    const void* depthRaw, size_t depthSize,
    const void* colorRaw, size_t colorSize)
>callback) {

    if (frameNeedsUpdate) {
        callback(
            p_depth_frame, width * height * sizeof(uint16_t),
            p_color_frame, INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char));

        frameNeedsUpdate = false;
        return true;
    }
    return false;
}