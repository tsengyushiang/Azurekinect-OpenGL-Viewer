#include "examples-pcl.h"


unsigned int* fast_triangulation_of_unordered_pcd(
	float* points,int count, int& indicesCount,
	float searchRadius,
	int maximumNearestNeighbors,
	float maximumSurfaceAngle
) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	
	cloud->width = count;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->resize(cloud->width * cloud->height);
	
	for (int i = 0; i < count; i++) {
		cloud->points[i].x = points[i * 6 + 0];
		cloud->points[i].y = points[i * 6 + 1];
		cloud->points[i].z = points[i * 6 + 2];
	}

	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	gp3.setSearchRadius(searchRadius);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(maximumNearestNeighbors);
	gp3.setMaximumSurfaceAngle(M_PI / 2);
	gp3.setMinimumAngle(M_PI / 18);
	gp3.setMaximumAngle(2 * M_PI / 3);
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	;

	unsigned int* indices = (unsigned int*)calloc(  triangles.polygons.size() * 3, sizeof(unsigned int));
	indicesCount = triangles.polygons.size() * 3;

	for (int i = 0; i < triangles.polygons.size(); i++) {
		indices[i * 3 + 0] = triangles.polygons[i].vertices[0];
		indices[i * 3 + 1] = triangles.polygons[i].vertices[1];
		indices[i * 3 + 2] = triangles.polygons[i].vertices[2];
	}
	return indices;
}

void fast_triangulation_of_unordered_pcd_fromFile(std::string filename= "bun0.pcd")
{
	// Load input file into a PointCloud<T> with an appropriate type
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCLPointCloud2 cloud_blob;
	pcl::io::loadPCDFile(filename, cloud_blob);
	pcl::fromPCLPointCloud2(cloud_blob, *cloud);
	//* the data should be available in cloud

	std::ofstream myfile("result.obj");

	for (auto p : cloud->points) {
		myfile << "v " << p.x << " " << p.y << " " << p.z << std::endl;
	}

	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	gp3.setSearchRadius(0.025);         
	gp3.setMu(2.5);        
	gp3.setMaximumNearestNeighbors(100); 
	gp3.setMaximumSurfaceAngle(M_PI / 4); 
	gp3.setMinimumAngle(M_PI / 18);   
	gp3.setMaximumAngle(2 * M_PI / 3);
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	for (auto p : triangles.polygons) {
		myfile << "f " << p.vertices[0] + 1 << " " << p.vertices[1] + 1 << " " << p.vertices[2] + 1 << std::endl;
	}
	myfile.close();
}