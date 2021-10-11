#pragma once
#include "./InputBase.h"
#include "../json/jsonUtils.h"

class JsonData :public InputBase {

    std::thread autoUpdate;
    void getLatestFrame();

public :

    std::vector<std::string> framefiles;

    int currentFrame;
    int frameLength;

    JsonData(int w,int h);
    ~JsonData();

    bool setFrameIndex(int) override;
    void updateFrame();
};