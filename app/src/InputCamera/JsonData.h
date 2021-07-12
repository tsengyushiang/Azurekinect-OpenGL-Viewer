#pragma once
#include "./InputBase.h"

class JsonData :public InputBase {

public :

    int currentFrame;
    int frameLength;

    JsonData(int w,int h):InputBase(w,h,w,h) {};
    bool fetchframes(std::function<void(
        const void* depthRaw, size_t depthSize,
        const void* colorRaw, size_t colorSize)
    >callback) override;
};