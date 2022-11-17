#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
#define PI 3.14159265358979323846264338327950288419716939937510
#define BLOCK 16		// block size = 16 x 16

//# define Filter_size 3

void gradient_computation(Mat input, int block_num, float** hist) {
	int x, y, xx, yy, i, j;
	int height = input.rows;
	int width = input.cols;
	int idx;
	float conv_x, conv_y;

	// image size = 64x128 -> 105 blocks, 9-bin histogram per block
	float min = 10000000, max = -1;
	float* mag = (float*)calloc(height * width, sizeof(float));
	float* dir = (float*)calloc(height * width, sizeof(float));
	float* deg = (float*)calloc(height * width, sizeof(float));

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
			dir[y * width + x] = atan2(conv_y, conv_x);
			
			deg[y * width + x] = dir[y * width + x] * 180 / PI;

			if (deg[y * width + x] < 0)	deg[y * width + x] += 180.0;
			/*if (deg[y * width + x] > 180) {		// 180 넘는 값이 있는지 확인해봄
				printf("deg : %f, atan2 : %f \t (%d, %d)\n", deg[y * width + x], dir[y * width + x], x, y);
			}*/
			if (max < mag[y * width + x])	max = mag[y * width + x];
			if (min > mag[y * width + x])	min = mag[y * width + x];
		}
	}

	int block_idx = 0;	// block counter

	// feature descriptor	
	for (y = 0; y <= height - BLOCK; y += BLOCK / 2) {
		for (x = 0; x <= width - BLOCK; x += BLOCK / 2) {
			for (yy = y; yy < y + BLOCK; yy++) {
				for (xx = x; xx < x + BLOCK; xx++) {
					idx = deg[yy * width + xx] / 20.0;	// Quantization
					hist[block_idx][idx] += mag[yy * width + xx];
				}
			}
			block_idx++;
		}
	}

	// Normalization
	float hist_nom = 0.0;
	x = 0;
	for (i = 0; i < block_idx; i++) {
		for (j = 0; j < 9; j++) {
			hist_nom += hist[i][j] * hist[i][j];
		}
		for (x = 0; x < 9; x++) {
			hist[i][x] /= sqrt(hist_nom);
		}
		hist_nom = 0;
	}

	//just for visualization
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			result.at<uchar>(y, x) = 255 * (mag[y * width + x] - min) / (max - min);
		}
	}
	imwrite("result.bmp", result);

	free(mag);
	free(dir);
	free(deg);
}

FILE* fp;

void compare_histogram(char * filename, float** ori, float** cmp, int block_num) {
	int i, j;

	fp = fopen(filename, "wt");

	for (i = 0; i < block_num; i++) {
		fprintf(fp, "%d번째 block, ", i + 1);
		for (j = 0; j < 9; j++) {
			fprintf(fp, "%f, ", ori[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n\n");

	for (i = 0; i < block_num; i++) {
		fprintf(fp, "%d번째 block, ", i + 1);
		for (j = 0; j < 9; j++) {
			fprintf(fp, "%f, ", cmp[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n\n");

	for (i = 0; i < block_num; i++) {
		fprintf(fp, "%d번째 block, ", i + 1);
		for (j = 0; j < 9; j++) {
			fprintf(fp, "%f, ", abs(ori[i][j] - cmp[i][j]));		// 원본 이미지와 비교 이미지와의 차이
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


void main() {
	int height, width, i;

	Mat imgGray = imread("lecture3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat cmp_img1 = imread("compare1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat cmp_img2 = imread("compare2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	height = imgGray.rows;
	width = imgGray.cols;

	int block_num = (width / (BLOCK / 2) - 1) * (height / (BLOCK / 2) - 1);		// 105 blocks 

	float** hist_ori = (float**)calloc(block_num, sizeof(float*));
	for (i = 0; i < block_num; i++) {
		hist_ori[i] = (float*)calloc(9, sizeof(float));		// 9-bin histogram
	}
	gradient_computation(imgGray, block_num, hist_ori);

	float** hist_cmp1 = (float**)calloc(block_num, sizeof(float*));
	for (i = 0; i < block_num; i++) {
		hist_cmp1[i] = (float*)calloc(9, sizeof(float));		
	}
	gradient_computation(cmp_img1, block_num, hist_cmp1);

	float** hist_cmp2 = (float**)calloc(block_num, sizeof(float*));
	for (i = 0; i < block_num; i++) {
		hist_cmp2[i] = (float*)calloc(9, sizeof(float));		
	}
	gradient_computation(cmp_img2, block_num, hist_cmp2);

	char filename1[] = "compare1.csv";
	compare_histogram(filename1, hist_ori, hist_cmp1, block_num);

	char filename2[] = "compare2.csv";
	compare_histogram(filename2, hist_ori, hist_cmp2, block_num);

	// 동적 할당 해제
	for (i = 0; i < block_num; i++) {
		free(hist_ori[i]);
	}
	free(hist_ori);

	for (i = 0; i < block_num; i++) {
		free(hist_cmp1[i]);
	}
	free(hist_cmp1);

	for (i = 0; i < block_num; i++) {
		free(hist_cmp2[i]);
	}
	free(hist_cmp2);
}