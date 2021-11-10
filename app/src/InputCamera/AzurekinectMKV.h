#pragma once

#include "./InputBase.h"

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <map>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

#include <opencv2/highgui/highgui.hpp>  // Video write

typedef struct MKVInfo {
    int w;
    int h;
    long long lastTimeStamp;
};

class AzureKinectMKV :public InputBase {
    int index = 0;
    long long currentTimeStamp = -1;
    uint32_t offset;
    k4a_record_configuration_t config;
    k4a::playback device_handle;
    k4a::capture capture;
    k4a::image depth_image;
    k4a::image color_image;
    k4a::image transformed_depth_image;
    k4a::calibration calibration;
    k4a::transformation transformation;

    std::thread autoUpdate;
    void getLatestFrame();

    std::string folder;
    bool exportMode = false;
    cv::VideoWriter* oVideoWriter = nullptr;

public:
    int frameLength;

    AzureKinectMKV(int w, int h, std::string device,bool exportMode);
    ~AzureKinectMKV();

    int currentFrame;
    bool setFrameIndex(int) override;
    void updateFrame();

    static MKVInfo open(std::string);

};