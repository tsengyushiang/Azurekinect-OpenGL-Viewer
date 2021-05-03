#include "opecv-utils.h"

std::vector<glm::vec2> OpenCVUtils::opencv_detect_aruco_from_RealsenseRaw(int w, int h, const uint16_t* p_depth_frame, const uchar* p_color_frame) {

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)p_color_frame, cv::Mat::AUTO_STEP);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<glm::vec2> corner2d;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);
    // if at least one marker detected
    if (ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(image, corners, ids);
        for (auto points : corners) {
            for (auto p : points) {
                corner2d.push_back(glm::vec2(p.x, p.y));
            }
        }       
    }
    //cv::imshow("Aruco", image);
    return corner2d;
}