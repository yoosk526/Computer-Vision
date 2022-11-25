#include "main.h"

void draw_rectangle(Mat input, int* cornerMap, int cornerNum, Mat mask_old) {
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float density;
	int th = 10;
	int i = 0;
	int min_x = 100000;
	int max_x = -1;
	int min_y = 100000;
	int max_y = -1;
	int* subMap = (int*)calloc(cornerNum * 2, sizeof(int));

	Mat rect_img;
	input.copyTo(rect_img);

	//For circle
	Scalar c;
	Point pCenter;
	int radius = 10;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (cornerMap[y * width + x] == 1) {
				density = compute_density(input, cornerMap, x, y, width, height);
				if (density > th) {
					subMap[i] = x;
					subMap[i + 1] = y;
					i += 2;
					if (x < min_x)	min_x = x;
					if (x > max_x)	max_x = x;
					if (y < min_y)	min_y = y;
					if (y > max_y)	max_y = y;
					pCenter.x = x;
					pCenter.y = y;
					c.val[1] = 0;
					c.val[1] = 0;
					c.val[2] = 255;
					circle(rect_img, pCenter, radius, c, -1);
				}
			}
		}
	}
	rectangle(rect_img, Rect(Point(min_x - rectRegion, min_y- rectRegion), Point(max_x + rectRegion, max_y + rectRegion)), Scalar(0, 255, 0), 2, 8, 0);

	imwrite("./images/rect_image.bmp", rect_img);
	imshow("rect image", rect_img);
	waitKey(2000);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (min_x - rectRegion <= x && x <= max_x + rectRegion && min_y - rectRegion <= y && y <= max_y + rectRegion)
				mask_old.at<uchar>(y, x) = 255;
		}
	}
	imwrite("./images/mask_img.bmp", mask_old);


	free(subMap);
}