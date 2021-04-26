// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-net/rs_net.hpp>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <thread>

void opencv_panels(std::string url) {

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    rs2::net_device dev(url);
    rs2::context ctx;
    dev.add_to(ctx);
    rs2::pipeline pipe(ctx);

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    // Start streaming with default recommended configuration
    pipe.start(cfg);

    using namespace cv;
    const auto window_name = "Depth Image" + url;
    const auto window_name2 = "Color Image" + url;

    namedWindow(window_name, WINDOW_FREERATIO);
    namedWindow(window_name2, WINDOW_FREERATIO);

    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame color = data.get_color_frame();

        // Query frame size (width and height)
        const int w = depth.as<rs2::video_frame>().get_width();
        const int h = depth.as<rs2::video_frame>().get_height();

        const int cw = color.as<rs2::video_frame>().get_width();
        const int ch = color.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
        Mat image2(Size(cw, ch), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

        // Update the window with new data
        imshow(window_name, image);
        imshow(window_name2, image2);
    }
}

int main(int argc, char* argv[]) try
{
    std::thread device1(opencv_panels, "192.168.0.120");
    std::thread device2(opencv_panels, "192.168.0.106");

    device1.join();
    device2.join();

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



