#include "main.h"

float compute_density(Mat input, int* cornerMap, int x, int y, int width, int height) {
	int yy, xx;
	float density;
	int numCorner = 0;

	for (yy = y - corner_box / 2; yy <= y + corner_box / 2; yy++) {
		for (xx = x - corner_box / 2; xx <= x + corner_box / 2; xx++) {
			if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
				if (cornerMap[yy * width + xx] == 1) {
					if (input.at<Vec3b>(yy, xx)[0] >= 255 - ErrRange && input.at<Vec3b>(yy, xx)[1] >= 255 - ErrRange && input.at<Vec3b>(yy, xx)[2] >= 255 - ErrRange) {
						numCorner++;
					}
				}
			}
		}
	}

	density = (numCorner / (corner_box * corner_box)) * 100;
	//printf("%f\n", density);

	return density;
}