#pragma once

#include <utility>      
#include <string>       
#include <iostream>     
#include <vector>
#include <Eigen/Core> 
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef struct PlaneEquation{
	Vector3d centroid;
	Vector3d plane_normal;
} PlaneEquation;
class EigenUtils {

public :

	static PlaneEquation best_plane_from_points(const std::vector<Vector3d>& c);

};