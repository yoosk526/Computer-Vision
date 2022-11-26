#include "main.h"

void graph_cut_mask(Mat input, Mat maskImg, Mat final_result) {
	int x, y;
	int height, width;

	height = input.rows;
	width = input.cols;

	Rect rect_range(0, 0, width - 1, height - 1);
	Mat mask = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

	Mat mask_result = Mat::zeros(height, width, CV_8UC1);
	Mat bg, fg;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (maskImg.at<Vec3b>(y, x)[0] == 255 && maskImg.at<Vec3b>(y, x)[1] == 0 && maskImg.at<Vec3b>(y, x)[2] == 0)
			{
				mask.at<uchar>(y, x) = GC_BGD;
			}
			else if (maskImg.at<Vec3b>(y, x)[0] == 0 && maskImg.at<Vec3b>(y, x)[1] == 0 && maskImg.at<Vec3b>(y, x)[2] == 255)
			{
				mask.at<uchar>(y, x) = GC_FGD;
			}
		}
	}

	final_result = input.clone();

	grabCut(input, mask, rect_range, bg, fg, 8, GC_INIT_WITH_MASK);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (mask.at<uchar>(y, x) == GC_FGD || mask.at<uchar>(y, x) == GC_PR_FGD) {
				mask_result.at<uchar>(y, x) = 255;
			}
			else {
				final_result.at<Vec3b>(y, x)[0] = 148;
				final_result.at<Vec3b>(y, x)[1] = 243;
				final_result.at<Vec3b>(y, x)[2] = 220;
			}
		}
	}
	imshow("graph_cut_result", final_result);
	imwrite("./images/mask_result.bmp", mask_result);
	imwrite("./images/graph_cut_result.bmp", final_result);
}