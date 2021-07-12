#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-net/rs_net.hpp>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/aruco.hpp>
#include <thread>
#include "./InputBase.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Realsense : public InputBase
{
    rs2::colorizer color_map;
    rs2_stream targetStream = RS2_STREAM_COLOR;
    rs2::align align_to_color{ targetStream };
    rs2::config cfg;

    rs2::pipeline* pipe;
    rs2::net_device* netdev=nullptr;

    float get_depth_scale(rs2::device dev);
public :
   
    Realsense(
        int cw,
        int ch,
        int dw,
        int dh
    );

    ~Realsense();

    bool opencvImshow = false;

    virtual void runDevice(std::string serial, rs2::context ctx);
    
    // CPU version get vertice (deprecated)
    //bool fetchframes(int pointcloudStride = 1);
    
    std::thread autoUpdate;
    void getLatestFrame();

    static void updateAvailableSerialnums(rs2::context& ctx);
    static std::set<std::string> availableSerialnums;
    static void HSVtoRGB(float H, float S, float V, uchar& R, uchar& G, uchar& B);
};

