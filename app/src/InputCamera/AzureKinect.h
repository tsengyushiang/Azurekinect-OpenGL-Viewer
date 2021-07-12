#pragma once

#include "./InputBase.h"
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
    AzureKinect(int w, int h);
    ~AzureKinect();

    void runDevice(std::string serailnum);

    std::thread autoUpdate;
    void getLatestFrame();

    static void updateAvailableSerialnums();
    static std::set<std::string> availableSerialnums;
    static std::map<std::string, k4a_device_t> availableDeviceDict;
};