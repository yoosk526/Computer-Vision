#include "main.h"

void graph_cut_mask(Mat input, Mat maskImg, Mat mask_result) {
	int x, y;
	int height, width;

	height = input.rows;
	width = input.cols;

	Rect rect_range(0, 0, width - 1, height - 1);
	Mat mask = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

	Mat final_result = Mat::zeros(height, width, CV_8UC3);
	Mat bg, fg;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (maskImg.at<Vec3b>(y, x)[0] == 0 && maskImg.at<Vec3b>(y, x)[1] == 0 && maskImg.at<Vec3b>(y, x)[2] == 255)
			{
				mask.at<uchar>(y, x) = GC_FGD;
			}
			else if ((maskImg.at<Vec3b>(y, x)[0] >= ErrRange || maskImg.at<Vec3b>(y, x)[1] >= ErrRange || maskImg.at<Vec3b>(y, x)[2] >= ErrRange) &&
				(maskImg.at<Vec3b>(y, x)[0] <= 255 - ErrRange || maskImg.at<Vec3b>(y, x)[1] <= 255 - ErrRange || maskImg.at<Vec3b>(y, x)[2] <= 255 - ErrRange))
			{
				mask.at<uchar>(y, x) = GC_BGD;
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
				final_result.at<Vec3b>(y, x)[0] = 0;
				final_result.at<Vec3b>(y, x)[1] = 0;
				final_result.at<Vec3b>(y, x)[2] = 0;
			}
		}
	}
	imshow("input", input);
	imshow("mask image", maskImg);
	imshow("mask result", mask_result);
	imshow("final result", final_result);
	waitKey(0);
	destroyAllWindows();

	imwrite("./images/grab_cut_result.bmp", final_result);
}