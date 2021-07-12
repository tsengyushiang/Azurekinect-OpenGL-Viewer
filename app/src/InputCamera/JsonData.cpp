#include "./JsonData.h"

bool JsonData::fetchframes(std::function<void(
    const void* depthRaw, size_t depthSize,
    const void* colorRaw, size_t colorSize)
>callback){

    callback(
        p_depth_frame + (currentFrame%frameLength) * width * height, width * height * sizeof(uint16_t),
        p_color_frame + (currentFrame % frameLength) * width * height * INPUT_COLOR_CHANNEL, INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char));

    return true;
}