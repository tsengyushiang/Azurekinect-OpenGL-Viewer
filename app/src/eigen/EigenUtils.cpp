#include "EigenUtils.h"

// return coefficients of plane through origin which means d in ax+by+cz+d=0 is zero.
PlaneEquation EigenUtils::best_plane_from_points(const std::vector<Vector3d>& c)
{
	//copy coordinates to  matrix in Eigen format
	size_t num_atoms = c.size();
	Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);
	for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

	//// calculate centroid
	Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

	//// subtract centroid
	coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

	//// we only need the left-singular matrix here
	////  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
	auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Vector3d plane_normal = svd.matrixU().rightCols<1>();
	return { centroid, plane_normal };
}