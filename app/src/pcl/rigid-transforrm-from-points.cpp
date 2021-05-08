#include "examples-pcl.h"

void pcl_rigid_transform_from_correspondPoints() {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>());

	cloud_in->width = 5;
	cloud_in->height = 1;
	cloud_in->is_dense = false;
	cloud_in->resize(cloud_in->width * cloud_in->height);

	cloud_out->width = 5;
	cloud_out->height = 1;
	cloud_out->is_dense = false;
	cloud_out->resize(cloud_out->width * cloud_out->height);

	cloud_in->points[0].x = 0;
	cloud_in->points[0].y = 0;
	cloud_in->points[0].z = 0;

	cloud_in->points[1].x = 2;
	cloud_in->points[1].y = 0;
	cloud_in->points[1].z = 0;

	cloud_in->points[2].x = 0;
	cloud_in->points[2].y = 2;
	cloud_in->points[2].z = 0;

	cloud_in->points[3].x = 3;
	cloud_in->points[3].y = 0;
	cloud_in->points[3].z = 4;

	cloud_in->points[4].x = 0;
	cloud_in->points[4].y = 3;
	cloud_in->points[4].z = 4;

	cloud_in->points[5].x = 0;
	cloud_in->points[5].y = 0;
	cloud_in->points[5].z = 5;

	// make a translation
	Eigen::Vector3f trans;
	trans << 0.5, 1.0, 0.75;
	//std::cout << "here" <<std::endl;
	std::vector<int> indices(cloud_in->points.size());
	for (int i = 0; i < cloud_in->points.size(); i++)
	{
		indices[i] = i;
		cloud_out->points[i].x = cloud_in->points[i].x + trans(0);
		cloud_out->points[i].y = cloud_in->points[i].y + trans(1);
		cloud_out->points[i].z = cloud_in->points[i].z + trans(2);
		std::cout << cloud_out->points[i].x << " ¡X " << cloud_out->points[i].y << " ¡X " << cloud_out->points[i].z << std::endl;
	}

	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation2;
	TESVD.estimateRigidTransformation(*cloud_in, *cloud_out, transformation2);
	std::cout << "The Estimated Rotation and translation matrices(using getTransformation function) are : \n" << std::endl;
	printf("\n");
	printf(" | % 6.3f % 6.3f % 6.3f | \n", transformation2(0, 0), transformation2(0, 1), transformation2(0, 2));
	printf("R = | % 6.3f % 6.3f % 6.3f | \n", transformation2(1, 0), transformation2(1, 1), transformation2(1, 2));
	printf(" | % 6.3f % 6.3f % 6.3f | \n", transformation2(2, 0), transformation2(2, 1), transformation2(2, 2));
	printf("\n");
	printf("t = < % 0.3f, % 0.3f, % 0.3f >\n", transformation2(0, 3), transformation2(1, 3), transformation2(2, 3));

	return ;
}