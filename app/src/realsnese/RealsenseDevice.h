#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-net/rs_net.hpp>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/aruco.hpp>
#include <thread>

typedef struct intrinsic {
    float fx;
    float fy;
    float ppx;
    float ppy;
    float depth_scale;

}Intrinsic;

class RealsenseDevice
{
    rs2::colorizer color_map;
    rs2::align align_to_color{ RS2_STREAM_COLOR };
    rs2::context ctx;
    rs2::config cfg;

    rs2::pipeline* pipe;
    rs2::net_device* dev;

    float get_depth_scale(rs2::device dev);
public :

    RealsenseDevice(int w = 640, int h = 480);
    ~RealsenseDevice();

    Intrinsic intri;

    int width;
    int height;
    const uint16_t* p_depth_frame;
    const uchar* p_color_frame;

    void runNetworkDevice(std::string url);
    bool fetchframes(bool opencvImshow=true);
};

