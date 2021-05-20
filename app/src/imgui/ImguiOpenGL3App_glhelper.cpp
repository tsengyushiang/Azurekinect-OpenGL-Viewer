#include "ImguiOpeGL3App.h"

void ImguiOpeGL3App::genCameraHelper(
	GLuint& vao, GLuint& vbo,
	float width,float height,float ppx, float ppy, float fx, float fy, // camera intrinsic,extrinsic 
	glm::ivec3 color,float size,bool isPlane // size and color
	) 
{
	std::vector<glm::vec2> uv = {
		glm::vec2(0.5,0.5),
		glm::vec2(0.0,0.0),
		glm::vec2(1.0,0.0),
		glm::vec2(1.0,1.0),
		glm::vec2(0.0,1.0)
	};

	std::vector<glm::vec4> points = {
		glm::vec4(ppx,ppy,isPlane? 1.0:0.0,1.0),
		glm::vec4(0,0,1.0,1.0),
		glm::vec4(width,0,1.0,1.0),
		glm::vec4(width,height,1.0,1.0),
		glm::vec4(0,height,1.0,1.0),
	};

	for (glm::vec4& p : points) {
		p = glm::vec4(
			(float(p.x) - ppx) / fx * size,
			(float(p.y) - ppy) / fy * size,
			p.z*size,
			1.0
		);
	}
	GLfloat frustum[6 * 3 * 4]; // 6 value * 3 vertex * 4 faces

	for (int i = 1; i < 5; i++) {

		int index1 = i;
		int index2 = 0;
		int index3 = (i + 1)>4?1:(i+1);

		// assign vertex and color
		frustum[6 * 3 * (i - 1) + 0] = points[index1].x;
		frustum[6 * 3 * (i - 1) + 1] = points[index1].y;
		frustum[6 * 3 * (i - 1) + 2] = points[index1].z;

		if (isPlane) {
			frustum[6 * 3 * (i - 1) + 3] = uv[index1].x;
			frustum[6 * 3 * (i - 1) + 4] = uv[index1].y;
			frustum[6 * 3 * (i - 1) + 5] = 1.0;
		}
		else {
			frustum[6 * 3 * (i - 1) + 3] = color.x;
			frustum[6 * 3 * (i - 1) + 4] = color.y;
			frustum[6 * 3 * (i - 1) + 5] = color.z;
		}


		frustum[6 * 3 * (i - 1) + 6] = points[index2].x;
		frustum[6 * 3 * (i - 1) + 7] = points[index2].y;
		frustum[6 * 3 * (i - 1) + 8] = points[index2].z;

		if (isPlane) {
			frustum[6 * 3 * (i - 1) + 9] = uv[index2].x;
			frustum[6 * 3 * (i - 1) + 10] = uv[index2].y;
			frustum[6 * 3 * (i - 1) + 11] = 1.0;
		}
		else {
			frustum[6 * 3 * (i - 1) + 9] = color.x;
			frustum[6 * 3 * (i - 1) + 10] = color.y;
			frustum[6 * 3 * (i - 1) + 11] = color.z;
		}


		frustum[6 * 3 * (i - 1) + 12] = points[index3].x;
		frustum[6 * 3 * (i - 1) + 13] = points[index3].y;
		frustum[6 * 3 * (i - 1) + 14] = points[index3].z;

		if (isPlane) {
			frustum[6 * 3 * (i - 1) + 15] = uv[index3].x;
			frustum[6 * 3 * (i - 1) + 16] = uv[index3].y;
			frustum[6 * 3 * (i - 1) + 17] = 1.0;
		}
		else {
			frustum[6 * 3 * (i - 1) + 15] = color.x;
			frustum[6 * 3 * (i - 1) + 16] = color.y;
			frustum[6 * 3 * (i - 1) + 17] = color.z;
		}
	}

	ImguiOpeGL3App::setPointsVAO(vao, vbo, frustum, 6 * 3 * 4);
}

void ImguiOpeGL3App::genOrigionAxis(GLuint& vao, GLuint& vbo) {
	//// draw xyz-axis
	GLfloat axisData[] = {
		//  X     Y     Z           R     G     B
			0.0f, 0.0f, 0.0f,       0.0f, 1.0f, 0.0f, // vertex 0
			0.0f, 0.1f, 0.0f,       0.0f, 1.0f, 0.0f, // vertex 1
			0.0f, 0.0f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 2
			0.1f, 0.0f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 3
			0.0f, 0.0f, 0.0f,       0.0f, 0.0f, 1.0f, // vertex 4
			0.0f, 0.0f, 0.1f,       0.0f, 0.0f, 1.0f, // vertex 5
	};
	ImguiOpeGL3App::setPointsVAO(vao, vbo, axisData, 6);
}