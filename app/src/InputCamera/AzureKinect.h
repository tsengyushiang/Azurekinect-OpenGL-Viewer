#pragma once

#include "./InputBase.h"
#include "MultiDeviceCapturer.h"
#include <k4a/k4a.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <map>

class AzureKinect :public InputBase {

    k4a_device_t device;
    k4a_device_configuration_t camera_configuration;
    k4a_calibration_t calibration;
    k4a_transformation_t transformation;
    k4a_image_t transformed_depth_image = NULL;


public:
    AzureKinect(int w=2048, int h=1536);
    ~AzureKinect();

    void runDevice(int index, bool isMaster);

    std::thread autoUpdate;
    void getLatestFrame();

    int indexOfMultiDeviceCapturer = -1;
    static MultiDeviceCapturer* capturer;
    static int alreadyStart;
    static void startDevices();

    static int ui_isMaster;
    static std::vector<std::string> availableSerialnums;
    static void updateAvailableSerialnums();
};