#include "main.h"

void main() {
	Mat img = imread("../../../media/img_sub.bmp", 1);
	int height = img.rows;
	int width = img.cols;
	resize(img, img, Size(width * 0.5, height * 0.5));
	
	// for corner detection
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	int* cornerMap = (int*)calloc(img.rows * img.cols, sizeof(int));
	int cornerNum = 0;

	// for inpaint
	height = img.rows;
	width = img.cols;

	Mat mask_old = Mat::zeros(height, width, CV_8UC1);

	// corner detection
	CornerDetection(img, img_gray, cornerMap, &cornerNum);

	// draw rectangle and make mask Image
	draw_rectangle(img, cornerMap, cornerNum, mask_old);

	// inpaint by rectangle mask
	Mat rect_result(height, width, CV_8UC3);
	inpaint(img, mask_old, rect_result, 3, INPAINT_TELEA);

	imshow("original", img);
	imshow("mask", mask_old);
	imshow("inpaint_by_Rect_mask_result", rect_result);
	waitKey(0);
	destroyAllWindows();

	imwrite("./images/inpaint_by_Rect_mask_result.bmp", rect_result);

	// inpaint by grab cut
	Mat maskImg = imread("./images/rect_image.bmp", 1);
	Mat mask_result = Mat::zeros(height, width, CV_8UC1);
	graph_cut_mask(img, maskImg, mask_result);

	Mat fin_result(height, width, CV_8UC3);
	inpaint(img, mask_result, fin_result, 3, INPAINT_TELEA);

	imshow("inpaint_by_mask_result", fin_result);
	waitKey(0);
	destroyAllWindows();

	imwrite("./images/inpaint_by_mask_result.bmp", fin_result);

	free(cornerMap);
}