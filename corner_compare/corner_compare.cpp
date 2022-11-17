#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>

LARGE_INTEGER freq, start, stop;
double duringtime;

#define win_size 3
#define PI 3.14159265358979323846264338327950288419716939937510

using namespace cv;

void CornerDetection(Mat input, Mat output, int* cornerMap, float th, int * corner_num, float* mag, float* deg) {
	int x, y, xx, yy;
	float conv_x, conv_y;
	float min = 1000000, max = -1;
	float IxIx, IyIy, IxIy, det, tr;
	float k = 0.04;
	int height = input.rows;	//input의 높이
	int width = input.cols;		//input의 너비

	float* dir = (float*)calloc(height * width, sizeof(float));

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
	
	Mat cornerness(height, width, CV_8UC1);		// gray scale

	int mask_size = 3;		// mask size = 3 x 3

	// Gradient Computation
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			conv_x = 0;
			conv_y = 0;

			//Inner product
			for (yy = y - mask_size / 2; yy <= y + mask_size / 2; yy++) {
				for (xx = x - mask_size / 2; xx <= x + mask_size / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						conv_x += input.at<uchar>(yy, xx) * mask_x[(yy - (y - mask_size / 2)) * mask_size + (xx - (x - mask_size / 2))];
						conv_y += input.at<uchar>(yy, xx) * mask_y[(yy - (y - mask_size / 2)) * mask_size + (xx - (x - mask_size / 2))];
					}
				}
			}

			Ix[y * width + x] = conv_x / 100.0;		// 값이 너무 크기 때문에 작게 만들어 줌
			Iy[y * width + x] = conv_y / 100.0;

			output.at<Vec3b>(y, x)[0] = input.at<uchar>(y, x);
			output.at<Vec3b>(y, x)[1] = input.at<uchar>(y, x);
			output.at<Vec3b>(y, x)[2] = input.at<uchar>(y, x);

			// gradient magnitude, phase computation
			conv_x /= 9.0;		
			conv_y /= 9.0;

			mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y);
			dir[y * width + x] = atan2(conv_y, conv_x);

			deg[y * width + x] = dir[y * width + x] * 180 / PI;

			if (deg[y * width + x] < 0)	deg[y * width + x] += 180.0;
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

			if (R[y * width + x] > th)
				cornerMap[y * width + x] = 1;

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
				(*corner_num)++;
			}

			cornerness.at<uchar>(y, x) = 255 - 255 * (R[y * width + x] - min) / (max - min);
		}
	}

	imwrite("./images/cornerMap.bmp", cornerness);

	free(Ix);
	free(Iy);
	free(R);
	free(dir);
}

void HoG_computation(int* cornerMap, int height, int width, float** hist, float* mag, float* deg) {
	int x, y, xx, yy, i, j;
	int idx;

	int corner_idx = 0;		// corner counter
	int b_size = 17;		// block size = 17 x 17

	// feature descriptor	
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (cornerMap[y * width + x] == 1) {
				for (yy = y - b_size / 2; yy <= y + b_size / 2; yy++) {
					for (xx = x - b_size / 2; xx <= x + b_size / 2; xx++) {
						if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
							idx = deg[yy * width + xx] / 20.0;	// Quantization
							hist[corner_idx][idx] += mag[yy * width + xx];
						}
					}
				}
				corner_idx++;
			}
		}
	}

	// Normalization
	float hist_nom = 0.0;
	x = 0;
	for (i = 0; i < corner_idx; i++) {
		for (j = 0; j < 9; j++) {
			hist_nom += hist[i][j] * hist[i][j];
		}
		for (x = 0; x < 9; x++) {
			hist[i][x] /= sqrt(hist_nom);
		}
		hist_nom = 0;
	}
}

void final_compare(Mat input, float** hist_ref, float** hist_tar, int* cornerMap_ref, int* cornerMap_tar) {
	int x, y, xx, yy, i, j;
	int height = input.rows;		// 원본 사이즈
	int width = input.cols / 2;

	int corner_idx_ref = 0;
	int corner_idx_tar = 0;

	float similarity = 0;
	float similarity_avg = 0;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (cornerMap_ref[y * width + x] == 1) {
				for (yy = 0; yy < height; yy++) {
					for (xx = 0; xx < width; xx++) {
						if (cornerMap_tar[yy * width + xx] == 1) {
							for (i = 0; i < 9; i++) {
								similarity = abs(hist_ref[corner_idx_ref][i] - hist_tar[corner_idx_tar][i]);
								similarity *= 100;
								similarity_avg = (similarity_avg * i + similarity) / (i + 1);
							}
							if (similarity_avg < 5) {		// 차이가 5%보다 작을 경우 유사하다고 판정
								line(input, Point(x, y), Point(xx + width, yy), Scalar(255, 0, 0), 2, 9, 0);
								cornerMap_tar[yy * width + xx] = 0;		// 이미 라인을 그린 위치는 다음 확인 때 통과하도록 하기 위해
								xx = width;
								yy = height;
								corner_idx_tar++;
							}

							// tar.bmp에만 있는 corner일 경우 통과해야 하므로
							if ((y * width + x) > (yy * width + xx)) {
								cornerMap_tar[yy * width + xx] = 0;	
								corner_idx_tar++;
							}
						}
						similarity = 0;
						similarity_avg = 0;
					}
				}
				corner_idx_ref++;
			}
		}
	}
  	imwrite("./images/final_result.bmp", input);
}

void main() {

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	int i, j, x, y;

	Mat ref = imread("ref.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat tar = imread("tar.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	float ref_th = 250, tar_th = 50;

	int height = ref.rows;
	int width = ref.cols;
	
	Mat output_ref(height, width, CV_8UC3);
	Mat output_tar(height, width, CV_8UC3);
	
	float* mag_ref = (float*)calloc(height * width, sizeof(float));
	float* deg_ref = (float*)calloc(height * width, sizeof(float));
	float* mag_tar = (float*)calloc(height * width, sizeof(float));
	float* deg_tar = (float*)calloc(height * width, sizeof(float));

	int* cornerMap_ref = (int*)calloc(height * width, sizeof(float));
	int* cornerMap_tar = (int*)calloc(height * width, sizeof(float));

	int ref_corner_num = 0;
	int tar_corner_num = 0;
	
	CornerDetection(ref, output_ref, cornerMap_ref, ref_th, &ref_corner_num, mag_ref, deg_ref);
	CornerDetection(tar, output_tar, cornerMap_tar, tar_th, &tar_corner_num, mag_tar, deg_tar);

	imwrite("./images/result_ref.bmp", output_ref);
	imwrite("./images/result_tar.bmp", output_tar);

	float** hist_ref = (float**)calloc(ref_corner_num, sizeof(float*));
	for (i = 0; i < ref_corner_num; i++) {
		hist_ref[i] = (float*)calloc(9, sizeof(float));		// 9-bin histogram
	}

	float** hist_tar = (float**)calloc(tar_corner_num, sizeof(float*));
	for (i = 0; i < tar_corner_num; i++) {
		hist_tar[i] = (float*)calloc(9, sizeof(float));		// 9-bin histogram
	}

	HoG_computation(cornerMap_ref, height, width, hist_ref, mag_ref, deg_ref);
	HoG_computation(cornerMap_tar, height, width, hist_tar, mag_tar, deg_tar);

	Mat line_result(height, width * 2, CV_8UC3);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width * 2; x++) {
			line_result.at<Vec3b>(y, x)[0] = (x < width) ? output_ref.at<Vec3b>(y, x)[0] : output_tar.at<Vec3b>(y, x - width)[0];
			line_result.at<Vec3b>(y, x)[1] = (x < width) ? output_ref.at<Vec3b>(y, x)[1] : output_tar.at<Vec3b>(y, x - width)[1];
			line_result.at<Vec3b>(y, x)[2] = (x < width) ? output_ref.at<Vec3b>(y, x)[2] : output_tar.at<Vec3b>(y, x - width)[2];
		}
	}

	final_compare(line_result, hist_ref, hist_tar, cornerMap_ref, cornerMap_tar);

	// 동적 할당 해제
	free(cornerMap_ref);
	free(cornerMap_tar);

	for (i = 0; i < ref_corner_num; i++) {
		free(hist_ref[i]);
	}
	free(hist_ref);

	for (i = 0; i < tar_corner_num; i++) {
		free(hist_tar[i]);
	}
	free(hist_tar);

	free(mag_ref);
	free(deg_ref);
	free(mag_tar);
	free(deg_tar);

	QueryPerformanceCounter(&stop);
	duringtime = (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart;
	printf("\nProcessing Time is %f (s)\n", duringtime);
}