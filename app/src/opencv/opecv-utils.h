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

private:

};