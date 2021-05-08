#pragma once

#include <iostream>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void pcl_rigid_transform_from_correspondPoints();

glm::mat4 pcl_pointset_rigid_calibrate(int size, std::vector<glm::vec3> srcPoint,std::vector<glm::vec3> dstPoints);
