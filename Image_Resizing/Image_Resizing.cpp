#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.141592

using namespace cv;

void ImageResize(Mat input, float scale) {
	int x, y;
	int height, width;			// 원본 이미지의 높이, 너비
	int re_height, re_width;	// resize된 이미지의 높이, 너비
	float pos_x, pos_y;			// Backward filling 사용한 원본 이미지의 좌표값
	int sx, sy;
	float f_x1, f_x2, f_y1, f_y2;	// Bi-linear interpolation 적용시 거리비
	float p1p3, p1p2, p2p4, p3p4;

	height = input.rows;
	width = input.cols;

	re_height = height * scale;
	re_width = width * scale;

	Mat result_NN(re_height, re_width, CV_8UC1);
	Mat result_BI(re_height, re_width, CV_8UC1);


	for (y = 0; y < re_height; y++) {
		for (x = 0; x < re_width; x++) {
			pos_x = (1.0 / scale) * x;
			pos_y = (1.0 / scale) * y;

			// Nearest Neighbor의 경우 0.5를 더하고 int
			sx = (int)(pos_x + 0.5);
			sy = (int)(pos_y + 0.5);
			result_NN.at<uchar>(y, x) = input.at<uchar>(sy, sx);

			// Bi linear
			sx = (int)pos_x;
			sy = (int)pos_y;
			// result_BI.at<uchar>(y, x) = 0.25 * (input.at<uchar>(sy, sx) + input.at<uchar>(sy, sx + 1) + input.at<uchar>(sy + 1, sx) + input.at<uchar>(sy + 1, sx + 1));
			f_x1 = pos_x - sx;	f_x2 = 1 - f_x1;
			f_y1 = pos_y - sy;	f_y2 = 1 - f_y1;
			
			p1p3 = input.at<uchar>(sy, sx) * f_y2 + input.at<uchar>(sy + 1, sx) * f_y1;
			p1p2 = input.at<uchar>(sy, sx) * f_x2 + input.at<uchar>(sy, sx + 1) * f_x1;
			p2p4 = input.at<uchar>(sy, sx + 1) * f_y2 + input.at<uchar>(sy + 1, sx + 1) * f_y1;
			p3p4 = input.at<uchar>(sy + 1, sx) * f_x2 + input.at<uchar>(sy + 1, sx + 1) * f_x1;

			result_BI.at<uchar>(y, x) = ((f_x2 * p1p3 + f_x1 * p2p4) + (f_y2 * p1p2 + f_y1 * p3p4)) / 2;
		}
	}

	imwrite("./images/resize_NN.jpg", result_NN);
	imwrite("./images/resize_BI.jpg", result_BI);
}


void ImageRotation(Mat input, float degree) {
	int x, y;
	int height, width;
	float pos_x, pos_y;		// resize 결과값 이미지 좌표값
	int sx, sy;
	float f_x1, f_x2, f_y1, f_y2;	// Bi-linear interpolation 적용시 거리비
	float p1p3, p1p2, p2p4, p3p4;
	float rad = degree * PI / 180.0;
	float R[2][2] = { {cos(rad), sin(rad)}, {-sin(rad), cos(rad)} };	// 역행렬

	height = input.rows;
	width = input.cols;

	Mat result_NN(height, width, CV_8UC1);
	Mat result_BI(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {		// y = 0 ~ (height - 1) 이므로
		for (x = 0; x < width; x++) {
			pos_x = R[0][0] * (x - width / 2) + R[0][1] * (y - height / 2);
			pos_y = R[1][0] * (x - width / 2) + R[1][1] * (y - height / 2);

			pos_x += width / 2;
			pos_y += height / 2;

			// Nearest Neighbor
			sx = (int)(pos_x + 0.5);
			sy = (int)(pos_y + 0.5);

			if (sx >= 0 && sx < width && sy >= 0 && sy < height)
				result_NN.at<uchar>(y, x) = input.at<uchar>(sy, sx);
			else
				result_NN.at<uchar>(y, x) = 0;

			// Bi linear
			sx = (int)pos_x;
			sy = (int)pos_y;

			if (sx >= 0 && sx < width - 1 && sy >= 0 && sy < height - 1) {	// 밑에서 +1 연산을 하기 때문에 범위 지정시 -1을 해줌
				// result_BI.at<uchar>(y, x) = 0.25 * (input.at<uchar>(sy, sx) + input.at<uchar>(sy, sx + 1) + input.at<uchar>(sy + 1, sx) + input.at<uchar>(sy + 1, sx + 1));
				f_x1 = pos_x - sx;	f_x2 = 1 - f_x1;
				f_y1 = pos_y - sy;	f_y2 = 1 - f_y1;
				p1p3 = input.at<uchar>(sy, sx) * f_y2 + input.at<uchar>(sy + 1, sx) * f_y1;
				p1p2 = input.at<uchar>(sy, sx) * f_x2 + input.at<uchar>(sy, sx + 1) * f_x1;
				p2p4 = input.at<uchar>(sy, sx + 1) * f_y2 + input.at<uchar>(sy + 1, sx + 1) * f_y1;
				p3p4 = input.at<uchar>(sy + 1, sx) * f_x2 + input.at<uchar>(sy + 1, sx + 1) * f_x1;
				result_BI.at<uchar>(y, x) = ((f_x2 * p1p3 + f_x1 * p2p4) + (f_y2 * p1p2 + f_y1 * p3p4)) / 2;
			}
			else
				result_BI.at<uchar>(y, x) = 0;
		}
	}

	imwrite("./images/rotate_NN.jpg", result_NN);
	imwrite("./images/rotate_BI.jpg", result_BI);
}

void main() {
	Mat imgGray = imread("C:/Users/yoosk/Documents/Images/car.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgGray = imread("edge_building.jpg", CV_LOAD_IMAGE_GRAYSCALE);


	printf("Image size : %d x %d (cols x rows)\n", imgGray.cols, imgGray.rows);

	float scale, degree;
	printf("Input your scale factor : ");
	scanf("%f", &scale);

	printf("Input your degree : ");
	scanf("%f", &degree);

	ImageResize(imgGray, scale);
	ImageRotation(imgGray, degree);
}