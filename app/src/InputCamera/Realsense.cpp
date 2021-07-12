#include "./Realsense.h"

void Realsense::HSVtoRGB(float H, float S, float V,uchar& R,uchar& G,uchar& B) {
    if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
        return;
    }
    float s = S / 100;
    float v = V / 100;
    float C = s * v;
    float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
    float m = v - C;
    float r, g, b;
    if (H >= 0 && H < 60) {
        r = C, g = X, b = 0;
    }
    else if (H >= 60 && H < 120) {
        r = X, g = C, b = 0;
    }
    else if (H >= 120 && H < 180) {
        r = 0, g = C, b = X;
    }
    else if (H >= 180 && H < 240) {
        r = 0, g = X, b = C;
    }
    else if (H >= 240 && H < 300) {
        r = X, g = 0, b = C;
    }
    else {
        r = C, g = 0, b = X;
    }
    R = (r + m) * 255;
    G = (g + m) * 255;
    B = (b + m) * 255;
}

std::set<std::string> Realsense::availableSerialnums;
void Realsense::updateAvailableSerialnums(rs2::context& ctx) {
    availableSerialnums.clear();
    for (auto&& dev : ctx.query_devices())
        availableSerialnums.insert(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
}

float Realsense::get_depth_scale(rs2::device dev)
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

Realsense::Realsense(int cw,int ch, int dw, int dh):InputBase(cw,ch,dw,dh)
{
}

Realsense::~Realsense() {
    pipe->stop();
    autoUpdate.join();
    cvDestroyWindow(serial.c_str());
    if (netdev!=nullptr)
        free(netdev);
}

void Realsense::runDevice(std::string serialnum, rs2::context ctx) {
    serial = serialnum;
    pipe = new rs2::pipeline(ctx);

    cfg.enable_stream(RS2_STREAM_DEPTH, dwidth, dheight, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, cwidth, cheight, RS2_FORMAT_BGRA8, 30);
    cfg.enable_device(serialnum);

    rs2::pipeline_profile profile = pipe->start(cfg);

    auto color_stream = profile.get_stream(targetStream).as<rs2::video_stream_profile>();
    
    auto intrins = color_stream.get_intrinsics();
    intri.fx = intrins.fx;
    intri.ppx = intrins.ppx;
    intri.fy = intrins.fy;
    intri.ppy = intrins.ppy;
    intri.depth_scale = get_depth_scale(profile.get_device());

    autoUpdate = std::thread(&Realsense::getLatestFrame, this);
}

void Realsense::getLatestFrame() {
    rs2::frameset frameset; // Wait for next set of frames from the camera
    while (true) {
        try {
            if (pipe->poll_for_frames(&frameset)) {
                rs2::frameset data = align_to_color.process(frameset);

                rs2::frame depth = data.get_depth_frame();
                rs2::frame color = data.get_color_frame();

                memcpy((void*)p_color_frame, color.get_data(), INPUT_COLOR_CHANNEL * width * height * sizeof(uchar));
                memcpy((void*)p_depth_frame, depth.get_data(), width * height * sizeof(uint16_t));
                frameNeedsUpdate = true;
            }
        }catch(...) {
            break;
        }            
    }   
}

// CPU version get vertice (deprecated)
//bool Realsense::fetchframes(int pointcloudStride) {
//
//    rs2::frameset frameset; // Wait for next set of frames from the camera
//
//    if (pipe->poll_for_frames(&frameset)) {
//        rs2::frameset data = align_to_color.process(frameset);
//
//        rs2::frame depth = data.get_depth_frame();
//        rs2::frame color = data.get_color_frame();
//
//        memcpy((void*)p_depth_frame, depth.get_data(), width * height * sizeof(uint16_t));
//        memcpy((void*)p_color_frame, color.get_data(), 3 * width * height * sizeof(uchar));
//
//        // copy to memory
//        vaildVeticesCount = 0;
//        for (int i = 0; i < height; i+= pointcloudStride) {
//            for (int j = 0; j < width; j+= pointcloudStride) {
//                int index = i * width + j;
//
//                glm::vec3 localPoint = colorPixel2point(glm::vec2(j, i));
//                if (localPoint.z > 0) {
//                    glm::vec4 point = modelMat * glm::vec4(
//                        localPoint.x,
//                        localPoint.y,
//                        localPoint.z,
//                        1.0
//                    );
//
//                    vertexData[vaildVeticesCount * 6 + 0] = point.x;
//                    vertexData[vaildVeticesCount * 6 + 1] = point.y;
//                    vertexData[vaildVeticesCount * 6 + 2] = point.z;
//
//                    vertexData[vaildVeticesCount * 6 + 3] = (float)p_color_frame[index * 3 + 2] / 255;
//                    vertexData[vaildVeticesCount * 6 + 4] = (float)p_color_frame[index * 3 + 1] / 255;
//                    vertexData[vaildVeticesCount * 6 + 5] = (float)p_color_frame[index * 3 + 0] / 255;
//                    vaildVeticesCount++;
//
//                    uchar R = 0;
//                    uchar G = 0;
//                    uchar B = 0;
//                    HSVtoRGB((localPoint.z / farPlane) * 360, 100, 100, R, G, B);
//                    p_depth_color_frame[index * 3 + 0] = R;
//                    p_depth_color_frame[index * 3 + 1] = G;
//                    p_depth_color_frame[index * 3 + 2] = B;
//                }
//                else {
//                    p_depth_color_frame[index * 3 + 0] = 0;
//                    p_depth_color_frame[index * 3 + 1] = 0;
//                    p_depth_color_frame[index * 3 + 2] = 0;
//                }
//            }
//        }
//
//        if (opencvImshow) {
//            rs2::frame depthColorize = depth.apply_filter(color_map);
//
//            const int w = depthColorize.as<rs2::video_frame>().get_width();
//            const int h = depthColorize.as<rs2::video_frame>().get_height();
//
//            const int cw = color.as<rs2::video_frame>().get_width();
//            const int ch = color.as<rs2::video_frame>().get_height();
//
//            cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depthColorize.get_data(), cv::Mat::AUTO_STEP);
//            cv::Mat image2(cv::Size(cw, ch), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
//
//            cv::Mat dst;
//            addWeighted(image, 0.5, image2, 0.5, 0.0, dst);
//            // Update the window with new data
//            imshow(serial.c_str(), dst);
//        }
//        else {
//            cvDestroyWindow(serial.c_str());
//        }
//        return true;
//    }
//    return false;
//}