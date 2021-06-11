#pragma once
#include "./RealsenseDevice.h"

class JsonRealsenseDevice :public RealsenseDevice {

public :
    JsonRealsenseDevice() {};
    bool fetchframes(std::function<void(
        const void* depthRaw, size_t depthSize,
        const void* colorRaw, size_t colorSize)
    >callback) override;
};