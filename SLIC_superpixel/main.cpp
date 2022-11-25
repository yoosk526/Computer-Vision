#include "main.h"

void main() {
	Mat image = imread("../../../media/edge_building_3.jpg", 1);
	resize(image, image, Size(image.cols * 0.5, image.rows * 0.5));

	SLICsegmentation(image);
}