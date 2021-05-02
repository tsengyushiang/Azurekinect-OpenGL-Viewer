#include "RealsenseDevice.h"

float RealsenseDevice::get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}
RealsenseDevice::RealsenseDevice(int w, int h ) {
    width = w;
    height = h;
    p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
    p_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
}

RealsenseDevice::~RealsenseDevice() {
    free((void*)p_depth_frame);
    free((void*)p_color_frame);
}

void RealsenseDevice::runNetworkDevice(std::string url) {
    // Declare depth colorizer for pretty visualization of depth data

    dev = new rs2::net_device(url);
    dev->add_to(ctx);
    pipe = new rs2::pipeline(ctx);


    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, 30);

    rs2::pipeline_profile profile = pipe->start(cfg);

    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    auto intrins = color_stream.get_intrinsics();
    intri.fx = intrins.fx;
    intri.ppx = intrins.ppx;
    intri.fy = intrins.fy;
    intri.ppy = intrins.ppy;
    intri.depth_scale = get_depth_scale(profile.get_device());
}

bool RealsenseDevice::fetchframes(bool opencvImshow) {

    rs2::frameset frameset; // Wait for next set of frames from the camera

    if (pipe->poll_for_frames(&frameset)) {
        rs2::frameset data = align_to_color.process(frameset);
      
        rs2::frame depth = data.get_depth_frame();
        rs2::frame color = data.get_color_frame();

        memcpy((void*)p_depth_frame, depth.get_data(), width * height * sizeof(uint16_t));
        memcpy((void*)p_color_frame, color.get_data(), 3 * width * height * sizeof(uchar));

        if (opencvImshow) {
            rs2::frame depthColorize = data.get_depth_frame().apply_filter(color_map);

            const int w = depthColorize.as<rs2::video_frame>().get_width();
            const int h = depthColorize.as<rs2::video_frame>().get_height();

            const int cw = color.as<rs2::video_frame>().get_width();
            const int ch = color.as<rs2::video_frame>().get_height();

            cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depthColorize.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat image2(cv::Size(cw, ch), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

            cv::Mat dst;
            addWeighted(image, 0.5, image2, 0.5, 0.0, dst);
            // Update the window with new data
            imshow("opencv", dst);
        }
        return true;
    }
    return false;
}