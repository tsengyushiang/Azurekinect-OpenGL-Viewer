#include "./examples-pcl.h"

glm::mat4 EigenToGlmMat(const Eigen::Matrix4f& v)
{
    glm::mat4 result;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result[i][j] = v(j, i);
        }
    }

    return result;
}

glm::mat4 pcl_icp(float correspondThreshold, std::vector<glm::vec3> sourcePoints, std::vector<glm::vec3> targetPoints
)
{
    //Creates two pcl::PointCloud<pcl::PointXYZ> boost shared pointers and initializes them
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the CloudIn data
    cloud_in->width = sourcePoints.size();
    cloud_in->height = 1;
    cloud_in->is_dense = false;
    cloud_in->points.resize(cloud_in->width * cloud_in->height);
    for (size_t i = 0; i < sourcePoints.size(); ++i)
    {
        cloud_in->points[i].x = sourcePoints[i].x;
        cloud_in->points[i].y = sourcePoints[i].y;
        cloud_in->points[i].z = sourcePoints[i].z;
    }

    cloud_out->width = targetPoints.size();
    cloud_out->height = 1;
    cloud_out->is_dense = false;
    cloud_out->points.resize(cloud_out->width * cloud_out->height);
    for (size_t i = 0; i < targetPoints.size(); ++i)
    {
        cloud_out->points[i].x = targetPoints[i].x;
        cloud_out->points[i].y = targetPoints[i].y;
        cloud_out->points[i].z = targetPoints[i].z;
    }

    //creates an instance of an IterativeClosestPoint and gives it some useful information
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);

    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance(correspondThreshold);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations(1);
    // Set the transformation epsilon (criterion 2)
    //icp.setTransformationEpsilon(1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    //icp.setEuclideanFitnessEpsilon(1);
    std::cout << "start icp..." << std::endl;
    //Creates a pcl::PointCloud<pcl::PointXYZ> to which the IterativeClosestPoint can save the resultant cloud after applying the algorithm
    pcl::PointCloud<pcl::PointXYZ> Final;

    //Call the registration algorithm which estimates the transformation and returns the transformed source (input) as output.
    icp.align(Final);

    //Return the state of convergence after the last align run. 
    //If the two PointClouds align correctly then icp.hasConverged() = 1 (true). 
    std::cout << "has converged: " << icp.hasConverged() << std::endl;

    //Obtain the Euclidean fitness score (e.g., sum of squared distances from the source to the target) 
    std::cout << "score: " << icp.getFitnessScore() << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    //Get the final transformation matrix estimated by the registration method. 
    auto transformation = icp.getFinalTransformation();
    std::cout << transformation << std::endl;   

    return EigenToGlmMat(transformation);
}