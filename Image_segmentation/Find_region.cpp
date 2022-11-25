#include "main.h"

Rect find_region(Mat input) {
	Mat image_rect;
	input.copyTo(image_rect);

	Rect point(10, 100, 380, 180);
	//Rect point(90, 80, 250, 160);
	//Rect point(1, 1, input.cols-1, input.rows-1);	// 전체 이미지
	rectangle(image_rect, point, Scalar(0, 0, 255), 2, 8, 0);

	imshow("image_rect", image_rect);
	waitKey(0);
	destroyAllWindows();

	return point;
}