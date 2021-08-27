#include "opecv-utils.h"

std::vector<glm::vec2> OpenCVUtils::getArucoMarkerConvexRegion(int w, int h, const uchar* p_color_frame, int colorChannel) {
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Mat image(cv::Size(w, h), colorChannel == 3 ? CV_8UC3 : CV_8UC4, (void*)p_color_frame, cv::Mat::AUTO_STEP);
    cv::Mat featurePoints= image.clone();

    if (colorChannel == 4) {
        cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<glm::vec2> corner2d;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);

    // if at least one marker detected
    if (ids.size() > 0) {
        for (auto points : corners) {
            for (auto p : points) {
                corner2d.push_back(glm::vec2(p.x, p.y));
            }
        }

        cv::Mat lines = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
        for (auto c1 : corner2d) {
            for (auto c2 : corner2d) {
                cv::Point2f p1(c1.x, c1.y);
                cv::Point2f p2(c2.x, c2.y);
                cv::line(lines, p1, p2, cv::Scalar(255, 255, 255));
            }
        }
        std::vector<std::vector<cv::Point> > contours; // Vector for storing contours
        cv::findContours(lines, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE); // Find the contours in the image
        int largest_area = 0;
        int largest_contour_index = 0;
        cv::Rect bounding_rect;
        for (size_t i = 0; i < contours.size(); i++) // iterate through each contour.
        {
            double area = contourArea(contours[i]);  //  Find the area of contour

            if (area > largest_area)
            {
                largest_area = area;
                largest_contour_index = i;               //Store the index of largest contour
                bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
            }
        }
        cv::Mat contour = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
        drawContours(contour, contours, largest_contour_index, cv::Scalar(255, 255, 255), -1); // Draw the largest contour using previously stored index.       

        corner2d.clear();
        for (int i = 0; i < contour.cols; ++i) {
            for (int j = 0; j < contour.rows; ++j) {
                if (contour.at<uchar>(j, i) > 0) {
                    featurePoints.at<cv::Vec4b>(j, i) = featurePoints.at<cv::Vec4b>(j, i)*0.5 + cv::Vec4b(255,0,0,255)*0.5;
                    corner2d.push_back(glm::vec2(i,j));
                }
            }
        }

        cv::imwrite("meta-floorCalibratefeaturePoints.png", featurePoints);
    }

    return corner2d;
}

std::vector<glm::vec2> OpenCVUtils::getArucoMarkerCorners(int w, int h,const uchar* p_color_frame, int colorChannel) {

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Mat image(cv::Size(w, h), colorChannel == 3 ? CV_8UC3 : CV_8UC4, (void*)p_color_frame, cv::Mat::AUTO_STEP);
    cv::Mat featurePoints = image.clone();
    if (colorChannel == 4) {
        cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<glm::vec2> corner2d;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);
    // if at least one marker detected
    if (ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(image, corners, ids);
        for (auto points : corners) {
            for (auto p : points) {
                cv::circle(featurePoints, p, 5, cv::Scalar(0, 0, 255,255),-1);
                corner2d.push_back(glm::vec2(p.x, p.y));
            }
        }       
    }
    cv::imwrite("meta-floorCalibrateCornerPoints.png", featurePoints);
    return corner2d;
}

void OpenCVUtils::saveMarkerBoard() {
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
    cv::Mat boardImage;
    board->draw(cv::Size(600, 500), boardImage, 10, 1);
    cv::imwrite("markboard.png", boardImage);
}