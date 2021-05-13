#include "RealsenseDevice.h"

std::set<std::string> RealsenseDevice::getAllDeviceSerialNumber(rs2::context& ctx) {
    std::set<std::string>              serials;
    for (auto&& dev : ctx.query_devices())
        serials.insert(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    return serials;
}

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
RealsenseDevice::RealsenseDevice() {

    if (targetStream == RS2_STREAM_COLOR) {
        width = cwidth;
        height = cheight;
    }
    else if (targetStream == RS2_STREAM_DEPTH) {
        width = dwidth;
        height = dheight;
    }

    p_depth_frame = (uint16_t*)calloc(width * height, sizeof(uint16_t));
    p_color_frame = (uchar*)calloc(3 * width * height, sizeof(uchar));
    vertexCount = width * height;
    vertexData = (float*)calloc(6 * vertexCount, sizeof(float)); // 6 represent xyzrgb

    modelMat = glm::mat4(1.0);
    //modelMat[3][0] = 1.0; // translate x
    //modelMat[3][1] = 0.0; // translate y
    //modelMat[3][2] = 0.0; // translate z
}

RealsenseDevice::~RealsenseDevice() {
    free((void*)p_depth_frame);
    free((void*)p_color_frame);
    pipe->stop();
    cvDestroyWindow(serial.c_str());
    if (netdev!=nullptr)
        free(netdev);
}

void RealsenseDevice::runDevice(std::string serialnum, rs2::context ctx) {
    serial = serialnum;
    pipe = new rs2::pipeline(ctx);

    cfg.enable_stream(RS2_STREAM_DEPTH, dwidth, dheight, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, cwidth, cheight, RS2_FORMAT_BGR8, 30);
    cfg.enable_device(serialnum);

    rs2::pipeline_profile profile = pipe->start(cfg);

    auto color_stream = profile.get_stream(targetStream).as<rs2::video_stream_profile>();
    
    auto intrins = color_stream.get_intrinsics();
    intri.fx = intrins.fx;
    intri.ppx = intrins.ppx;
    intri.fy = intrins.fy;
    intri.ppy = intrins.ppy;
    intri.depth_scale = get_depth_scale(profile.get_device());
}

std::string RealsenseDevice::runNetworkDevice(std::string url, rs2::context ctx) {
    // Declare depth colorizer for pretty visualization of depth data

    netdev = new rs2::net_device(url);
    netdev->add_to(ctx);
    pipe = new rs2::pipeline(ctx);

    serial = url;

    cfg.enable_stream(RS2_STREAM_DEPTH, dwidth, dheight, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, cwidth, cheight, RS2_FORMAT_BGR8, 30);

    rs2::pipeline_profile profile = pipe->start(cfg);

    auto color_stream = profile.get_stream(targetStream).as<rs2::video_stream_profile>();

    auto intrins = color_stream.get_intrinsics();
    intri.fx = intrins.fx;
    intri.ppx = intrins.ppx;
    intri.fy = intrins.fy;
    intri.ppy = intrins.ppy;
    intri.depth_scale = get_depth_scale(profile.get_device());

    return netdev->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
}

glm::vec3 RealsenseDevice::colorPixel2point(glm::vec2 pixel) {
    int i = pixel.y;
    int j = pixel.x;
    int index = i * width + j;

    float depthValue = (float)p_depth_frame[index] * intri.depth_scale;
    if (depthValue > farPlane) {
        return glm::vec3(0, 0, 0);
    }
    glm::vec3 point(
        (float(j) - intri.ppx) / intri.fx * depthValue,
        (float(i) - intri.ppy) / intri.fy * depthValue,
        depthValue
    );

    return point;
}


bool RealsenseDevice::fetchframes() {

    rs2::frameset frameset; // Wait for next set of frames from the camera

    if (pipe->poll_for_frames(&frameset)) {
        rs2::frameset data = align_to_color.process(frameset);

        rs2::frame depth = data.get_depth_frame();
        rs2::frame color = data.get_color_frame();

        memcpy((void*)p_depth_frame, depth.get_data(), width * height * sizeof(uint16_t));
        memcpy((void*)p_color_frame, color.get_data(), 3 * width * height * sizeof(uchar));

        // copy to memory
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;

                glm::vec3 point = colorPixel2point(glm::vec2(j, i));
                vertexData[index * 6 + 0] = point.x;
                vertexData[index * 6 + 1] = point.y;
                vertexData[index * 6 + 2] = point.z;

                vertexData[index * 6 + 3] = (float)p_color_frame[index * 3 + 2] / 255;
                vertexData[index * 6 + 4] = (float)p_color_frame[index * 3 + 1] / 255;
                vertexData[index * 6 + 5] = (float)p_color_frame[index * 3 + 0] / 255;
            }
        }

        if (opencvImshow) {
            rs2::frame depthColorize = depth.apply_filter(color_map);

            const int w = depthColorize.as<rs2::video_frame>().get_width();
            const int h = depthColorize.as<rs2::video_frame>().get_height();

            const int cw = color.as<rs2::video_frame>().get_width();
            const int ch = color.as<rs2::video_frame>().get_height();

            cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depthColorize.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat image2(cv::Size(cw, ch), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

            cv::Mat dst;
            addWeighted(image, 0.5, image2, 0.5, 0.0, dst);
            // Update the window with new data
            imshow(serial.c_str(), dst);
        }
        else {
            cvDestroyWindow(serial.c_str());
        }
        return true;
    }
    return false;
}