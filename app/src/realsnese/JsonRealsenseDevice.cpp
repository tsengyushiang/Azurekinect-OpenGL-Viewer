#include "JsonRealsenseDevice.h"

bool JsonRealsenseDevice::fetchframes(std::function<void(
    const void* depthRaw, size_t depthSize,
    const void* colorRaw, size_t colorSize)
>callback){
    
    callback(
        p_depth_frame, width * height * sizeof(uint16_t),
        p_color_frame, 3 * width * height * sizeof(uchar));
    
    return true;
}