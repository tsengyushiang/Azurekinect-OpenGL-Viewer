#include "AzureKinect.h"

/*
 This is utility to that provides query serial number and open device by specify serial number.

 // query serial numbers from all connected devices
 const std::vector<std::string> serial_numbers = k4a::device_query();
 if( serial_numbers.empty() ){
	 throw k4a::error( "devices not found!" );
 }

 // open device by specify serial number
 k4a::device device = k4a::device_open( serial_number[0] ); // specify string retrieved from k4a::device_query()
 //k4a::device device = k4a::device_open( "000000000000" ); // specify 12-digits string directly
 if( !device.is_valid() ){
	 throw k4a::error( "failed open device!" );
 }

 Copyright (c) 2020 Tsukasa Sugiura <t.sugiura0204@gmail.com>
 Licensed under the MIT license.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#ifndef __DEVICE_UTILITY__
#define __DEVICE_UTILITY__

#include <string>
#include <vector>

#include <k4a/k4a.hpp>

namespace k4a
{
	std::vector<std::string> device_query()
	{
		const int32_t device_count = k4a::device::get_installed_count();
		if (device_count == 0) {
			return std::vector<std::string>();
		}

		std::vector<std::string> serial_numbers;
		for (int32_t device_index = 0; device_index < device_count; device_index++) {
			try {
				k4a::device device = k4a::device::open(device_index);
				serial_numbers.push_back(device.get_serialnum());
				device.close();
			}
			catch (const k4a::error& error) {
				continue;
			}
		}

		return serial_numbers;
	}

	k4a::device device_open(const std::string serial_number)
	{
		constexpr int32_t length = 12;
		if (serial_number.empty() || serial_number.length() != length) {
			return k4a::device(nullptr);
		}

		const int32_t device_count = k4a::device::get_installed_count();
		if (device_count == 0) {
			return k4a::device(nullptr);
		}

		for (int32_t device_index = 0; device_index < device_count; device_index++) {
			try {
				k4a::device device = k4a::device::open(device_index);
				if (serial_number == device.get_serialnum()) return device;
				device.close();
			}
			catch (const k4a::error& error) {
				continue;
			}
		}

		return k4a::device(nullptr);
	}
}

#endif // __DEVICE_UTILITY__

int AzureKinect::ui_isMaster=0;
int AzureKinect::ui_Resolution=1;
const glm::vec2 AzureKinect::resolution[]{
	glm::vec2(0,0),
	glm::vec2(1280,720),
	glm::vec2(1920,1080),
	glm::vec2(2560,1440),
	glm::vec2(2048,1536),
};
std::vector<std::string> AzureKinect::availableSerialnums;
void AzureKinect::updateAvailableSerialnums() {
	availableSerialnums = k4a::device_query();
}

constexpr uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;
k4a_device_configuration_t AzureKinect::get_default_config()
{
	k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	camera_config.color_resolution = (k4a_color_resolution_t)AzureKinect::ui_Resolution;
	camera_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // No need for depth during calibration
	camera_config.camera_fps = K4A_FRAMES_PER_SECOND_30;     // Don't use all USB bandwidth
	camera_config.subordinate_delay_off_master_usec = 0;     // Must be zero for master
	camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
	return camera_config;
}

k4a_device_configuration_t get_master_config()
{
	k4a_device_configuration_t camera_config = AzureKinect::get_default_config();
	//camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

	// Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
	// sensor doesn't interfere with the other. To accomplish this the master depth image captures
	// (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
	// depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
	// images centered around the color image as closely as possible.
	camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
	return camera_config;
}

// Subordinate customizable settings
k4a_device_configuration_t get_subordinate_config()
{
	k4a_device_configuration_t camera_config = AzureKinect::get_default_config();
	//camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

	// Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
	// sensor doesn't interfere with the other. To accomplish this the master depth image captures
	// (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
	// depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
	// images centered around the color image as closely as possible.
	camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
	return camera_config;
}

AzureKinect::AzureKinect(int w, int h) :InputBase(w,h,w,h){
};

void AzureKinect::setXY_table() {

	k4a_float2_t p;
	k4a_float3_t ray;
	int valid;

	for (int y = 0; y < height; y++)
	{
		p.xy.y = (float)y;
		for (int x = 0; x < width; x++)
		{
			int idx = (height -1 -y) * width + x;
			p.xy.x = (float)x;

			k4a_calibration_2d_to_3d(
				&calibration, &p, 1.f, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &ray, &valid);

			if (valid)
			{
				xy_table[idx*2] = ray.xyz.x;
				xy_table[idx*2+1] = ray.xyz.y;
			}
			else
			{
				xy_table[idx * 2] = 0;
				xy_table[idx * 2 + 1] = 0;
			}
		}
	}
	cudaMemcpy(xy_table_cuda, xy_table, width * height * 2 * sizeof(float), cudaMemcpyHostToDevice);
	xy_tableReady = true;
}


void AzureKinect::runDevice(int index,bool isMaster) {

	serial = AzureKinect::availableSerialnums[index];

	if (K4A_FAILED(k4a_device_open(index, &device)))
	{
		printf("Failed to open k4a device %d!\n", index);
		return;
	}

	camera_configuration = isMaster ? get_master_config() : get_subordinate_config();
	k4a_device_start_cameras(device, &camera_configuration);
	k4a_device_get_calibration(device, camera_configuration.depth_mode, camera_configuration.color_resolution, &calibration);
	transformation = k4a_transformation_create(&calibration);

	//aligned depth container
	if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
		calibration.color_camera_calibration.resolution_width,
		calibration.color_camera_calibration.resolution_height,
		calibration.color_camera_calibration.resolution_width * (int)sizeof(uint16_t),
		&transformed_depth_image))
	{
		printf("Failed to create transformed depth image\n");
	}

	intri.fx = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
	intri.fy = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
	intri.ppx = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
	intri.ppy = calibration.color_camera_calibration.intrinsics.parameters.param.cy;

	// TODO : get the actual scale value!!
	intri.depth_scale = 1e-3;
	setXY_table();

	autoUpdate = std::thread(&AzureKinect::getLatestFrame, this);
}

AzureKinect::~AzureKinect() {
	autoUpdate.join();
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
			if (!enableUpdateFrame) continue;
			k4a_capture_t capture;
			k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE);
			if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
			{
				k4a_image_t colorImage = k4a_capture_get_color_image(capture);

				if (colorImage != nullptr)
				{
					if (!enableUpdateFrame) continue;
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
					if (!enableUpdateFrame) continue;
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
