#pragma once

#include <thread>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define INPUT_COLOR_CHANNEL 4

typedef struct intrinsic {
    float fx;
    float fy;
    float ppx;
    float ppy;
    float depth_scale;

}Intrinsic;

class InputBase
{

public:

    InputBase(
        int cw,
        int ch,
        int dw,
        int dh
    );

    ~InputBase();

    glm::mat4 modelMat;
    bool calibrated = false;
    Intrinsic intri;
    std::string serial;

    // result resolution of aligned depth/color
    int width;
    int height;

    // resolution for color/depth
    int cwidth, cheight, dwidth, dheight;

    uint16_t* p_depth_frame;
    unsigned char* p_color_frame;
    float farPlane = 5.0;

    bool frameNeedsUpdate = false;
    virtual bool fetchframes(std::function<void(
        const void* depthRaw, size_t depthSize,
        const void* colorRaw, size_t colorSize)
    >callback);

    // fetch signle pixel-point
    glm::vec3 colorPixel2point(glm::vec2);
};

