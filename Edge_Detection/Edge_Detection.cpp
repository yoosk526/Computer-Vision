#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
#define PI 3.141592
#define BLK 16

//# define Filter_size 3

void gradient_computation(Mat input) {
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y, deg;
	
	float* hist = (float*)calloc(9, sizeof(float));		// histogram
	int idx;

	float min = 10000000, max = -1;
	float* mag = (float*)calloc(height * width, sizeof(float));
	float* dir = (float*)calloc(height * width, sizeof(float));

	/*int mask_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	int mask_y[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };*/
	int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	Mat result(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			conv_x = 0;
			conv_y = 0;

			//inner product
			for (yy = y - 1; yy <= y + 1; yy++) {
				for (xx = x - 1; xx <= x + 1; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						conv_x += input.at<uchar>(yy, xx) * mask_x[(yy - (y - 1)) * 3 + xx - (x - 1)];
						conv_y += input.at<uchar>(yy, xx) * mask_y[(yy - (y - 1)) * 3 + xx - (x - 1)];
					}
				}
			}
			//크기를 줄여줌 (필요한 연산 x)
			conv_x /= 9.0;
			conv_y /= 9.0;

			mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y);
			//dir = atan2(conv_y, conv_x);
			//deg = dir * 180 / PI;
			dir[y * width + x] = atan2(conv_y, conv_x);
			deg = dir[y * width + x] * 180 / PI;

			if (deg < 0)	deg += 180.0;

			idx = deg / 20.0;	// Quantization
			hist[idx] += mag[y * width + x];

			if (max < mag[y * width + x])	max = mag[y * width + x];
			if (min > mag[y * width + x])	min = mag[y * width + x];
		}
	}
	
	
	for (int i = 0; i < 9; i++) {
		printf("%d: %f\n", i, hist[i] / (height * width));		// 값이 너무 커서 그냥 h * w 로 나눔
	}
	
	//just for visualization
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			result.at<uchar>(y, x) = 255 * (mag[y * width + x] - min) / (max - min);
		}
	}
	imwrite("result.bmp", result);
	imshow("result", result);
	waitKey(0);
	destroyAllWindows()

	free(mag);
	free(dir);
	free(hist);
}


void main() {
	int height, width;
	Mat imgGray = imread("../../../media/car.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	height = imgGray.rows;
	width = imgGray.cols;

	//Mat imgEdge(height, width, CV_8UC1);

	//Magnitude(imgGray, imgEdge, height, width);

	gradient_computation(imgGray);

}