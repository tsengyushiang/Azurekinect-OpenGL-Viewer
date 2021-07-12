#include "AzureKinect.h"

std::map<std::string, k4a_device_t> AzureKinect::availableDeviceDict;
std::set<std::string> AzureKinect::availableSerialnums;
void AzureKinect::updateAvailableSerialnums() {
	//clean up
    availableSerialnums.clear();
	for (const auto& deiveDict : availableDeviceDict) {
		k4a_device_t device = deiveDict.second;
		k4a_device_close(device);
	}
	availableDeviceDict.clear();

	//get avalible device
	uint32_t count = k4a_device_get_installed_count();
	for (uint32_t i = 0; i < count; i++) {
		// Open the first plugged in Kinect device
		k4a_device_t device = NULL;
		if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device)))
		{
			printf("Failed to open k4a device %d!\n",i);
		}

		// Get the size of the serial number
		size_t serial_size = 0;
		k4a_device_get_serialnum(device, NULL, &serial_size);

		// Allocate memory for the serial, then acquire it
		char* serial = (char*)(malloc(serial_size));
		k4a_device_get_serialnum(device, serial, &serial_size);

		availableDeviceDict[std::string(serial)] = device;
		availableSerialnums.insert(std::string(serial));
		
		free(serial);
	}
}

AzureKinect::AzureKinect(int w, int h) :InputBase(w, h, w, h),device(nullptr) {
	camera_configuration = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	if (w == 1920 && h == 1080) {
		camera_configuration.color_resolution = K4A_COLOR_RESOLUTION_1080P;
	}
	else if (w == 1280 && h == 720) {
		camera_configuration.color_resolution = K4A_COLOR_RESOLUTION_720P;
	}
	camera_configuration.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	camera_configuration.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	camera_configuration.camera_fps = K4A_FRAMES_PER_SECOND_30;
};


void AzureKinect::runDevice(std::string serailnum) {
	device = AzureKinect::availableDeviceDict[serailnum];
	serial = serailnum;
	k4a_device_start_cameras(device, &camera_configuration);

	k4a_device_get_calibration(device, camera_configuration.depth_mode, camera_configuration.color_resolution, &calibration);
	transformation = k4a_transformation_create(&calibration);

	intri.fx = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
	intri.fy = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
	intri.ppx = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
	intri.ppy = calibration.color_camera_calibration.intrinsics.parameters.param.cy;

	// TODO : get the actual scale value!!
	intri.depth_scale = 1e-3;

	//aligned depth container
	if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
		calibration.color_camera_calibration.resolution_width,
		calibration.color_camera_calibration.resolution_height,
		calibration.color_camera_calibration.resolution_width * (int)sizeof(uint16_t),
		&transformed_depth_image))
	{
		printf("Failed to create transformed depth image\n");
	}

	autoUpdate = std::thread(&AzureKinect::getLatestFrame, this);
}

AzureKinect::~AzureKinect() {
	k4a_device_stop_cameras(device);
	autoUpdate.join();
	k4a_image_release(transformed_depth_image);
}


void AzureKinect::getLatestFrame() {


	while (true) {
		try {
			k4a_capture_t capture;
			k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE);
			if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
			{
				k4a_image_t colorImage = k4a_capture_get_color_image(capture);

				if (colorImage != nullptr)
				{
					memcpy((void*)p_color_frame, k4a_image_get_buffer(colorImage), INPUT_COLOR_CHANNEL * width * height * sizeof(unsigned char));
					k4a_image_release(colorImage);
					frameNeedsUpdate = true;
				}

				k4a_image_t depthImage = k4a_capture_get_depth_image(capture);
				if (depthImage != nullptr)
				{
					if (K4A_RESULT_SUCCEEDED !=
						k4a_transformation_depth_image_to_color_camera(transformation,
							depthImage,
							transformed_depth_image))
					{
						printf("Failed to compute transformed depth image\n");
					}

					memcpy((void*)p_depth_frame, (uint16_t*)(void*)k4a_image_get_buffer(transformed_depth_image), width * height * sizeof(uint16_t));

					k4a_image_release(depthImage);
					frameNeedsUpdate = true;
				}
			}
			k4a_capture_release(capture);
		}
		catch (...) {
			break;
		}
	}	
}