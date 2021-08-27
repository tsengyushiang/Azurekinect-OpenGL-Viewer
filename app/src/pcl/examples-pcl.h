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

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h> // for KdTree
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

glm::mat4 pcl_icp(float correspondThreshold, std::vector<glm::vec3> sourcePoints, std::vector<glm::vec3> targetPoints);

void pcl_rigid_transform_from_correspondPoints();

glm::mat4 pcl_pointset_rigid_calibrate(int size, std::vector<glm::vec3> srcPoint,std::vector<glm::vec3> dstPoints);
unsigned int* fast_triangulation_of_unordered_pcd(
	float* points, int count, int& indicesCount,
	float searchRadius=0.1,
	int maximumNearestNeighbors=30,
	float maximumSurfaceAngle = M_PI/4
);