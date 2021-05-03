#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-net/rs_net.hpp>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/aruco.hpp>
#include <thread>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
    rs2::config cfg;

    rs2::pipeline* pipe;
    rs2::net_device* dev;

    float get_depth_scale(rs2::device dev);
public :

    glm::mat4 modelMat;

    RealsenseDevice(int w = 640, int h = 480);
    ~RealsenseDevice();

    bool calibrated=false;
    bool opencvImshow = false;

    Intrinsic intri;

    std::string serial;
    int width;
    int height;

    const uint16_t* p_depth_frame;
    const uchar* p_color_frame;

    int vertexCount;
    float* vertexData = nullptr;

    void runNetworkDevice(std::string url, rs2::context ctx);
    void runDevice(std::string serial, rs2::context ctx);
    bool fetchframes();
    glm::vec3 colorPixel2point(glm::vec2);

    static std::set<std::string> getAllDeviceSerialNumber(rs2::context& ctx);
};

