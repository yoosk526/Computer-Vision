#include "main.h"

void main() {
	Mat image = imread("../../../media/truck.jpg");

	// find region of rectangle
	Rect point = find_region(image);

	// Graph cut by rectangle region
	graph_cut(image, point);
 
	// Graph cut by mask image
	Mat maskImg = imread("../../../media/truck_mask.bmp", 1);
	graph_cut_mask(image, maskImg);

} 