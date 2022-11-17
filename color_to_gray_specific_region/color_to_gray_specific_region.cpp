#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;

void main() {
	Mat imgColor = imread("new.jpg", CV_LOAD_IMAGE_COLOR);
	Mat imgBoundary = imgColor.clone();

	printf("Image size : %d x %d\n", imgColor.cols, imgColor.rows);


	
	int cx, cy, w;	// Boundary의 센터 값
	printf("Input center position, width : ");
	scanf("%d %d %d", &cx, &cy, &w);

	int rVal, gVal, bVal, gray;

	for (int y = cy - w; y <= cy + w; y++) {
		for (int x = cx - w; x <= cx + w; x++) {
			bVal = imgColor.at<Vec3b>(y, x)[0];
			gVal = imgColor.at<Vec3b>(y, x)[1];
			rVal = imgColor.at<Vec3b>(y, x)[2];
			gray = (bVal + gVal + rVal) / 3;
			imgBoundary.at<Vec3b>(y, x)[0] = gray;
			imgBoundary.at<Vec3b>(y, x)[1] = gray;
			imgBoundary.at<Vec3b>(y, x)[2] = gray;
		}
	}

	imshow("Boundary", imgBoundary);

	waitKey(5000);
	/*
	
	int y = 100;
	int x = 100;
	printf("%d %d %d\n", imgColor.at<Vec3b>(y, x)[0], imgColor.at<Vec3b>(y, x)[1], imgColor.at<Vec3b>(y, x)[2]);
	
	for (y = 0; y < imgColor.rows; y++) {
		for (x = 0; x < imgColor.cols; x++) {
			if (imgColor.at<Vec3b>(y, x)[0] == 192 && imgColor.at<Vec3b>(y, x)[1] == 216 && imgColor.at<Vec3b>(y, x)[2] == 236) {
				imgBoundary.at<Vec3b>(y, x)[0] = 255;
				imgBoundary.at<Vec3b>(y, x)[1] = 255;
				imgBoundary.at<Vec3b>(y, x)[2] = 255;
			}
		}
	}
	imshow("Boundary", imgBoundary);
	*/

}