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
    rs2_stream targetStream = RS2_STREAM_COLOR;
    rs2::align align_to_color{ targetStream };
    rs2::config cfg;

    rs2::pipeline* pipe;
    rs2::net_device* netdev=nullptr;

    float get_depth_scale(rs2::device dev);
public :

    glm::mat4 modelMat;

    RealsenseDevice();
    ~RealsenseDevice();

    bool visible = true;
    bool calibrated=false;
    bool opencvImshow = false;

    Intrinsic intri;

    std::string serial;

    // result resolution depend on targetStream var
    int width;
    int height;

    // resolution for color/depth
    int cwidth=1280;
    int cheight=720;
    int dwidth = 1280;
    int dheight = 720;

    const uint16_t* p_depth_frame;
    uchar* p_depth_color_frame;
    const uchar* p_color_frame;
    float farPlane = 5.0;

    int vaildVeticesCount = 0;
    int vertexCount;
    float* vertexData = nullptr;

    std::string runNetworkDevice(std::string url, rs2::context ctx);
    void runDevice(std::string serial, rs2::context ctx);
    bool fetchframes(int pointcloudStride = 1);
    bool fetchframes(std::function<void (
        const void* depthRaw, size_t depthSize,
        const void* colorRaw, size_t colorSize)
    >callback);
    glm::vec3 colorPixel2point(glm::vec2);

    static std::set<std::string> getAllDeviceSerialNumber(rs2::context& ctx);
    static void HSVtoRGB(float H, float S, float V, uchar& R, uchar& G, uchar& B);
};

