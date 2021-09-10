#pragma once

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/aruco.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OpenCVUtils
{
public:

	static void saveMarkerBoard();
	static std::vector<glm::vec2> getArucoMarkerCorners(int w,int h, const uchar* p_color_frame, int colorChannel);
	
	// calculate by line every 2 points and find contours
	static std::vector<glm::vec2> getArucoMarkerConvexRegion(int w, int h, const uchar* p_color_frame, int colorChannel);

	static std::map<std::string, std::pair<cv::Point2f, cv::Point2f>> getCorrespondingArucoMarker(
		int w1, int h1, const uchar* p_color_frame1, int frameType1,
		int w2, int h2, const uchar* p_color_frame2, int frameType2, bool centerOnly = true);

private:

	static cv::Ptr<cv::aruco::Dictionary> dictionary;

};