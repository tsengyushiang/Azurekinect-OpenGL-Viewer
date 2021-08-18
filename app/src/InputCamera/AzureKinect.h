#pragma once

#include "./InputBase.h"
#include "MultiDeviceCapturer.h"
#include <k4a/k4a.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <map>

class AzureKinect :public InputBase {

    k4a::calibration calibration;
    k4a::transformation* depth_to_color;

public:
    AzureKinect(int w=1280, int h=720);
    ~AzureKinect();

    void runDevice(int index);

    std::thread autoUpdate;
    void getLatestFrame();

    int indexOfMultiDeviceCapturer = -1;
    static MultiDeviceCapturer* capturer;
    static bool alreadyStart;
    static void startDevices();
};