#include "./AzureKinectMKV.h"
#include "../json/jsonUtils.h"
#include <opencv2/core/utils/filesystem.hpp>

AzureKinectMKV::AzureKinectMKV(int w, int h, std::string filename,bool exportMode) :InputBase(w, h, w, h), exportMode(exportMode){
	device_handle = k4a::playback::open(filename.c_str());
	config = device_handle.get_record_configuration();
	frameLength = device_handle.get_recording_length().count();
	calibration = device_handle.get_calibration();
	transformation = k4a::transformation(calibration);
	device_handle.get_tag("K4A_DEVICE_SERIAL_NUMBER", &(this->serial));
	device_handle.set_color_conversion(K4A_IMAGE_FORMAT_COLOR_BGRA32);
	intri.fx = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
	intri.fy = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
	intri.ppx = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
	intri.ppy = calibration.color_camera_calibration.intrinsics.parameters.param.cy;
	offset = config.start_timestamp_offset_usec;
	std::cout << offset << " " << w << " " << h << " " << intri.fx << " " << intri.fy << " " << intri.ppx << " " << intri.ppy << std::endl;
	// TODO : get the actual scale value!!
	intri.depth_scale = 1e-3;
	setXYtable(intri.ppx, intri.ppy, intri.fx, intri.fy);

	transformed_depth_image = k4a::image::create(
		K4A_IMAGE_FORMAT_DEPTH16,
		calibration.color_camera_calibration.resolution_width,
		calibration.color_camera_calibration.resolution_height,
		calibration.color_camera_calibration.resolution_width * (int)sizeof(uint16_t));

	// set start frame
	//device_handle.seek_timestamp(std::chrono::microseconds(offset), K4A_PLAYBACK_SEEK_BEGIN);
	if (exportMode) {
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		std::ostringstream sstr;
		sstr << "./" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
		folder = sstr.str();
		cv::utils::fs::createDirectory(folder);
		cv::Size frame_size(width, height);
		oVideoWriter = new  cv::VideoWriter(cv::utils::fs::join(folder, serial)+".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
			30, frame_size, true);
	}

	autoUpdate = std::thread(&AzureKinectMKV::getLatestFrame, this);
};

AzureKinectMKV::~AzureKinectMKV() {
	autoUpdate.join();
}

bool AzureKinectMKV::setFrameIndex(int index) {
	syncTime = index+ offset;
	try {
		//device_handle.seek_timestamp(std::chrono::microseconds(syncTime), K4A_PLAYBACK_SEEK_BEGIN);
		updateFrame();
		return true;
	}catch (...) {
		std::cout << "failed to seek timestamp " << syncTime << std::endl;
		return false;
	}
}

void AzureKinectMKV::updateFrame() {
	k4a::capture capture = NULL;
	if (device_handle.get_next_capture(&capture)) {

		color_image = capture.get_color_image();
		if (currentTime == color_image.get_device_timestamp().count()) {
			return;
		}

		currentTime = color_image.get_device_timestamp().count();
		if (color_image) {
			memcpy((void*)p_color_frame, color_image.get_buffer(), INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char));
			/*cv::Mat image(cv::Size(width, height), CV_8UC4, (void*)color_image.get_buffer(), cv::Mat::AUTO_STEP);
			cv::imwrite(serial+"_"+std::to_string(color_image.get_device_timestamp().count())+".png", image);*/
			std::cout << serial << " get color frame at timestamp : " << color_image.get_device_timestamp().count() << std::endl;
			frameNeedsUpdate = true;
		}
		else {
			std::cout<<serial<<" failed to get color frame at timestamp : " << syncTime <<std::endl;
			return;
		}

		depth_image = capture.get_depth_image();
		if (depth_image)
		{
			transformation.depth_image_to_color_camera(depth_image, &transformed_depth_image);
			memcpy((void*)p_depth_frame, (uint16_t*)transformed_depth_image.get_buffer(), width * height * sizeof(uint16_t));
			frameNeedsUpdate = true;
		}
		++index;

		if(exportMode){
			cv::Mat image(cv::Size(width, height), INPUT_COLOR_CHANNEL == 4 ? CV_8UC4 : CV_8UC3, (void*)p_color_frame, cv::Mat::AUTO_STEP);
			cv::cvtColor(image, image, CV_BGRA2RGBA);
			oVideoWriter->write(image);
			/*std::ostringstream stringStream;
			stringStream << "./" << serial << "_t" << std::setw(8) << std::setfill('0') << std::to_string(index);
			std::string filename = cv::utils::fs::join(folder, stringStream.str());
			JsonUtils::saveRealsenseJson(filename,
				width, height,
				intri.fx, intri.fy, intri.ppx, intri.ppy,
				intri.depth_scale, p_depth_frame, p_color_frame, xy_table,
				farPlane, {
					esitmatePlaneCenter.x,
					esitmatePlaneCenter.y,
					esitmatePlaneCenter.z,
					esitmatePlaneNormal.x,
					esitmatePlaneNormal.y,
					esitmatePlaneNormal.z,
					point2floorDistance
				}
			);*/
		}		
	}
	else if (exportMode) {
		if (oVideoWriter != NULL) {
			delete oVideoWriter;
			oVideoWriter = nullptr;
		}
	}
}

void AzureKinectMKV::getLatestFrame() {
	while (true)
	{
		if (!enableUpdateFrame) continue;
		updateFrame();
	}

}

//https://docs.microsoft.com/zh-tw/azure/kinect-dk/record-playback-api
MKVInfo AzureKinectMKV::open(std::string filename) {
	auto device_handle = k4a::playback::open(filename.c_str());
	auto config = device_handle.get_record_configuration();
	int w = 0, h = 0;
	switch (config.color_resolution) {
		case K4A_COLOR_RESOLUTION_OFF:     /**< Color camera will be turned off with this setting */
			w = 0; h = 0;	break;
		case K4A_COLOR_RESOLUTION_720P:    /**< 1280 * 720  16:9 */
			w = 1280; h = 720; break;
		case K4A_COLOR_RESOLUTION_1080P:   /**< 1920 * 1080 16:9 */
			w = 1920; h = 1080; break;
		case K4A_COLOR_RESOLUTION_1440P:   /**< 2560 * 1440 16:9 */
			w = 2560; h = 1440; break;
		case K4A_COLOR_RESOLUTION_1536P:   /**< 2048 * 1536 4:3  */
			w = 2048; h = 1536; break;
		case K4A_COLOR_RESOLUTION_2160P:   /**< 3840 * 2160 16:9 */
			w = 3840; h = 2160; break;
	}
	std::string serialnum;
	device_handle.get_tag("K4A_DEVICE_SERIAL_NUMBER", &serialnum);
	int len = device_handle.get_recording_length().count();
	std::cout << serialnum<< " "<< config.start_timestamp_offset_usec<<" "<< len << std::endl;

	device_handle.close();
	return {
		w,
		h,
		len 
	};
}