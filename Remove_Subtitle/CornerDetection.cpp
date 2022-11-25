#include "main.h"

void CornerDetection(Mat input, Mat input_gray, int* cornerMap, int* cornerNum) {
	int x, y, xx, yy;
	float conv_x, conv_y;
	float min = 1000000, max = -1;
	float IxIx, IyIy, IxIy, det, tr;
	float k = 0.04, th = 100;		// th = 40000000000;
	int height = input.rows;		// input의 높이
	int width = input.cols;			// input의 너비

	//Filter for 2-D convolution
	int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	//For circle
	Scalar c;
	Point pCenter;
	int radius = 3;

	float* Ix = (float*)calloc(height * width, sizeof(float));
	float* Iy = (float*)calloc(height * width, sizeof(float));
	float* R = (float*)calloc(height * width, sizeof(float));
	

	Mat output(height, width, CV_8UC3);			// output image
	Mat cornerness(height, width, CV_8UC1);		// gray scale

	int b_size = 3;

	// Gradient Computation
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			conv_x = 0;
			conv_y = 0;

			//Inner product
			for (yy = y - b_size / 2; yy <= y + b_size / 2; yy++) {
				for (xx = x - b_size / 2; xx <= x + b_size / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						conv_x += input_gray.at<uchar>(yy, xx) * mask_x[(yy - (y - b_size / 2)) * b_size + (xx - (x - b_size / 2))];
						conv_y += input_gray.at<uchar>(yy, xx) * mask_y[(yy - (y - b_size / 2)) * b_size + (xx - (x - b_size / 2))];
					}
				}
			}
			Ix[y * width + x] = conv_x / 100.0;		// 값이 너무 크기 때문에 작게 만들어 줌
			Iy[y * width + x] = conv_y / 100.0;


			output.at<Vec3b>(y, x)[0] = input.at<Vec3b>(y, x)[0];
			output.at<Vec3b>(y, x)[1] = input.at<Vec3b>(y, x)[1];
			output.at<Vec3b>(y, x)[2] = input.at<Vec3b>(y, x)[2];
		}
	}

	//R computation
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			IxIx = 0;
			IyIy = 0;
			IxIy = 0;

			for (yy = y - win_size / 2; yy <= y + win_size / 2; yy++) {
				for (xx = x - win_size / 2; xx <= x + win_size / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						IxIx += Ix[yy * width + xx] * Ix[yy * width + xx];
						IyIy += Iy[yy * width + xx] * Iy[yy * width + xx];
						IxIy += Ix[yy * width + xx] * Iy[yy * width + xx];
					}
				}
			}

			det = IxIx * IyIy - IxIy * IxIy;
			tr = IxIx + IyIy;
			R[y * width + x] = det - k * tr * tr;

			if (R[y * width + x] > th) {
				cornerMap[y * width + x] = 1;
				(*cornerNum)++;
			}

			if (R[y * width + x] > max) max = R[y * width + x];
			if (R[y * width + x] < min) min = R[y * width + x];

		}
	}

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (cornerMap[y * width + x] == 1) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[1] = 0;
				c.val[1] = 255;
				c.val[2] = 0;
				circle(output, pCenter, radius, c, 1, 8, 0);
			}
			cornerness.at<uchar>(y, x) = 255 - 255 * (R[y * width + x] - min) / (max - min);
		}
	}

	imwrite("./images/result.bmp", output);
	imwrite("./images/cornerMap.bmp", cornerness);
	imshow("corner output", output);
	waitKey(2000);

	free(Ix);
	free(Iy);
	free(R);
}