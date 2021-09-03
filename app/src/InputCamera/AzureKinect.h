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
    AzureKinect(int w, int h);
    ~AzureKinect();

    void setXY_table();
    void runDevice(int index, bool isMaster);

    std::thread autoUpdate;
    void getLatestFrame();       
    
    static int ui_isMaster;
    static int ui_Resolution;
    static const std::string items[];
    static const glm::vec2 resolution[];
    static std::vector<std::string> availableSerialnums;
    static k4a_device_configuration_t get_default_config();
    static void updateAvailableSerialnums();
};