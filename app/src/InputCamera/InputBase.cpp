#include "InputBase.h"

InputBase::InputBase(int cw, int ch, int dw, int dh)
    :cwidth(cw), cheight(ch), dwidth(dw), dheight(dh)
{
    width = cwidth;
    height = cheight;

    p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
    p_color_frame = (unsigned char*)calloc(INPUT_COLOR_CHANNEL * width * height, sizeof(unsigned char));

    modelMat = glm::mat4(1.0);
}

InputBase::~InputBase() {
    free((void*)p_depth_frame);
    free((void*)p_color_frame);
}

glm::vec3 InputBase::colorPixel2point(glm::vec2 pixel) {
    int i = pixel.y;
    int j = pixel.x;
    int index = i * width + j;

    float depthValue = (float)p_depth_frame[index] * intri.depth_scale;
    if (depthValue > farPlane) {
        return glm::vec3(0, 0, 0);
    }
    glm::vec3 point(
        (float(j) - intri.ppx) / intri.fx * depthValue,
        (float(i) - intri.ppy) / intri.fy * depthValue,
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