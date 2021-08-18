#include "AzureKinect.h"

constexpr uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;
k4a_device_configuration_t get_default_config()
{
	k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	camera_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // No need for depth during calibration
	camera_config.camera_fps = K4A_FRAMES_PER_SECOND_30;     // Don't use all USB bandwidth
	camera_config.subordinate_delay_off_master_usec = 0;     // Must be zero for master
	camera_config.synchronized_images_only = true;
	return camera_config;
}

k4a_device_configuration_t get_master_config()
{
	k4a_device_configuration_t camera_config = get_default_config();
	camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

	// Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
	// sensor doesn't interfere with the other. To accomplish this the master depth image captures
	// (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
	// depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
	// images centered around the color image as closely as possible.
	camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
	camera_config.synchronized_images_only = true;
	return camera_config;
}

// Subordinate customizable settings
k4a_device_configuration_t get_subordinate_config()
{
	k4a_device_configuration_t camera_config = get_default_config();
	camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

	// Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
	// sensor doesn't interfere with the other. To accomplish this the master depth image captures
	// (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
	// depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
	// images centered around the color image as closely as possible.
	camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
	return camera_config;
}

bool AzureKinect::alreadyStart = false;
MultiDeviceCapturer* AzureKinect::capturer;
void AzureKinect::startDevices() {
	int32_t color_exposure_usec = 3e+4;  // somewhat reasonable default exposure time
	int32_t powerline_freq = 2;          // default to a 60 Hz powerline
	size_t num_devices = k4a::device::get_installed_count();
	std::vector<uint32_t> device_indices;
	for (int i = 0; i < num_devices; i++) {
		device_indices.emplace_back(i);
	}
	try {
		capturer = new MultiDeviceCapturer(device_indices, color_exposure_usec, powerline_freq);
		// Create configurations for devices
		k4a_device_configuration_t main_config = get_master_config();
		if (num_devices == 1) // no need to have a master cable if it's standalone
		{
			main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
		}
		k4a_device_configuration_t secondary_config = get_subordinate_config();
		capturer->start_devices(main_config, secondary_config);
	}
	catch (...) {
		printf("Azure kinect device canonot open.");
	}
}

AzureKinect::AzureKinect(int w, int h) :InputBase(w, h, w, h){
};


void AzureKinect::runDevice(int index) {

	indexOfMultiDeviceCapturer = index;

	if (indexOfMultiDeviceCapturer ==0) {
		serial = capturer->get_master_device().get_serialnum();
		auto config = get_master_config();
		calibration = capturer->get_master_device().get_calibration(config.depth_mode, config.color_resolution);
	}
	else if(indexOfMultiDeviceCapturer>0){
		serial = capturer->get_subordinate_device_by_index(indexOfMultiDeviceCapturer - 1).get_serialnum();
		auto config = get_subordinate_config();
		calibration = capturer->get_subordinate_device_by_index(indexOfMultiDeviceCapturer-1).get_calibration(config.depth_mode, config.color_resolution);
	}
	depth_to_color = new k4a::transformation(calibration);

	intri.fx = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
	intri.fy = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
	intri.ppx = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
	intri.ppy = calibration.color_camera_calibration.intrinsics.parameters.param.cy;

	// TODO : get the actual scale value!!
	intri.depth_scale = 1e-3;

	autoUpdate = std::thread(&AzureKinect::getLatestFrame, this);
}

AzureKinect::~AzureKinect() {
	autoUpdate.join();
	free(depth_to_color);
}

k4a::image create_depth_image_like(const k4a::image& im)
{
	return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
		im.get_width_pixels(),
		im.get_height_pixels(),
		im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

void AzureKinect::getLatestFrame() {


	while (true) {
		try {
			auto captures = capturer->get_synchronized_captures(get_subordinate_config(), true);
			auto capture = captures[indexOfMultiDeviceCapturer];
			k4a::image main_color_image  = capture.get_color_image();
			k4a::image main_depth_image = capture.get_depth_image();
			k4a::image main_aligned_depth = create_depth_image_like(main_color_image);
			depth_to_color->depth_image_to_color_camera(main_depth_image, &main_aligned_depth);

			if (main_color_image.is_valid())
			{
				memcpy((void*)p_color_frame, main_color_image.get_buffer(), INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char));
				frameNeedsUpdate = true;
			}

			if (main_aligned_depth.is_valid())
			{
				memcpy((void*)p_depth_frame, (uint16_t*)(void*)main_aligned_depth.get_buffer(), width * height * sizeof(uint16_t));
				frameNeedsUpdate = true;
			}
		}
		catch (...) {
			break;
		}
	}	
}