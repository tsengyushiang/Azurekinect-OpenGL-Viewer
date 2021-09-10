#include "opecv-utils.h"

typedef struct CorrespondingAruco {
    std::vector<cv::Point2f> srccorners;
    std::vector<cv::Point2f> trgcorners;
};

cv::Ptr<cv::aruco::Dictionary> OpenCVUtils::dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

std::map<std::string, std::pair<cv::Point2f, cv::Point2f>> OpenCVUtils::getCorrespondingArucoMarker(
    int w1, int h1, const uchar* p_color_frame1, int colorChannel1,
    int w2, int h2, const uchar* p_color_frame2, int colorChannel2, bool centerOnly) {

    cv::Mat imageSrc(cv::Size(w1, h1), colorChannel1 == 3 ? CV_8UC3 : CV_8UC4, (void*)p_color_frame1, cv::Mat::AUTO_STEP);
    cv::Mat imageTrg(cv::Size(w2, h2), colorChannel2 == 3 ? CV_8UC3 : CV_8UC4, (void*)p_color_frame2, cv::Mat::AUTO_STEP);

    // detect marker
    cv::Mat imageSrcGray, imageTrgGray;
    if (colorChannel1 == 4) {
        cvtColor(imageSrc, imageSrcGray, cv::COLOR_BGRA2GRAY);
        cvtColor(imageSrc, imageSrc, cv::COLOR_BGRA2BGR);
    }
    if (colorChannel2 == 4) {
        cvtColor(imageTrg, imageTrgGray, cv::COLOR_BGRA2GRAY);
        cvtColor(imageTrg, imageTrg, cv::COLOR_BGRA2BGR);
    }

    std::vector<int> srcids, trgids;
    std::vector<std::vector<cv::Point2f> > srccorners, trgcorners;
    cv::aruco::detectMarkers(imageSrcGray, dictionary, srccorners, srcids);
    cv::aruco::detectMarkers(imageTrgGray, dictionary, trgcorners, trgids);

    /*
        map<id_cornerId,<sorucePoint,targetPoint>>
        id_cornerId : X_0,X_1,X_2,X_3
    */
    std::map<std::string, std::pair<cv::Point2f, cv::Point2f>> correspondingPoints;
    for (int i = 0; i < srccorners.size(); i++) {
        int id = srcids[i];
        std::vector<cv::Point2f>& corners = srccorners[i];        
        for (int j = 0; j < corners.size(); ++j) {
            correspondingPoints[std::to_string(id) + "_" + std::to_string(j)] = 
                std::pair<cv::Point2f, cv::Point2f>(cv::Point2f(-1,-1),cv::Point2f(-1,-1));
            correspondingPoints[std::to_string(id) + "_" + std::to_string(j)].first = corners[j];
        }
    }
    for (int i = 0; i < trgcorners.size(); i++) {
        int id = trgids[i];
        std::vector<cv::Point2f>& corners = trgcorners[i];
        for (int j = 0; j < corners.size(); ++j) {
            if (correspondingPoints.find(std::to_string(id) + "_" + std::to_string(j))!= correspondingPoints.end()) {
                correspondingPoints[std::to_string(id) + "_" + std::to_string(j)].second = corners[j];
            }
        }
    }

    {
        srand(time(NULL));
        cv::Mat correspondingPointMap(cv::Size(w1 + w2, std::max(h1, h2)), CV_8UC3);
        imageSrc.copyTo(correspondingPointMap(cv::Rect(0, 0, w1, h1)));
        imageTrg.copyTo(correspondingPointMap(cv::Rect(w1, 0, w2, h2)));
        for (auto keyvalue : correspondingPoints) {
            auto pointPair = keyvalue.second;

            cv::line(correspondingPointMap, pointPair.first, pointPair.second + cv::Point2f(w1, 0), cvScalar(
                int((double)rand() / (RAND_MAX + 1.0) * 255), 
                int((double)rand() / (RAND_MAX + 1.0) * 255), 
                int((double)rand() / (RAND_MAX + 1.0) * 255)), 1);
        }
        cv::imwrite("meta-CorrespondingArucoMarker.png", correspondingPointMap);
    }

    return correspondingPoints;
}

std::vector<glm::vec2> OpenCVUtils::getArucoMarkerConvexRegion(int w, int h, const uchar* p_color_frame, int colorChannel) {
    cv::Mat image(cv::Size(w, h), colorChannel == 3 ? CV_8UC3 : CV_8UC4, (void*)p_color_frame, cv::Mat::AUTO_STEP);
    cv::Mat featurePoints= image.clone();

    if (colorChannel == 4) {
        cvtColor(image, image, cv::COLOR_BGRA2GRAY);
        cvtColor(featurePoints, featurePoints, cv::COLOR_BGRA2GRAY);
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<glm::vec2> corner2d;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);

    // if at least one marker detected
    if (ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(featurePoints, corners, ids);
        cv::imwrite("meta-detectMarker.png", featurePoints);

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
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(4, 5, 0.15, 0.035, dictionary);
    cv::Mat boardImage;
    board->draw(cv::Size(5940*2, 8410*2), boardImage, 10, 1);
    cv::imwrite("markboard.png", boardImage);
}