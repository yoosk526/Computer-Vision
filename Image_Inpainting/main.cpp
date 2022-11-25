#include "main.h"

void main() {
	int height, width, x, y;
	
	Mat img = imread("../../../media/ref.bmp", 1);
	Mat mask = imread("../../../media/ref_mask.bmp", 1);

	height = img.rows;
	width = img.cols;

	Mat mask_New(height, width, CV_8UC1);
	Mat result(height, width, CV_8UC3);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (mask.at<Vec3b>(y, x)[0] == 255 && mask.at<Vec3b>(y, x)[1] == 255 && mask.at<Vec3b>(y, x)[2] == 255)
				mask_New.at<uchar>(y, x) = 255;
			else
				mask_New.at<uchar>(y, x) = 0;
		}
	}

	inpaint(img, mask_New, result, 3, INPAINT_TELEA);

	imshow("original", img);
	imshow("maskImg", mask);
	imshow("mask", mask_New);
	imshow("result", result);

	waitKey(0);
	destroyAllWindows();

	imwrite("result.bmp", result);
}