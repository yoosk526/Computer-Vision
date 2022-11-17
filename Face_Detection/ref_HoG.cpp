#include "main.h"

void ref_HoG(Mat input, float* refHOG_descriptor) {
	int x, y, xx, yy, i;
	int height = input.rows;
	int width = input.cols;
	//int BLK = width / 3;		// Block size = 12 x 12
	int idx;
	float conv_x, conv_y;
	float hist_nom;				// Normalization 할 때 임시로 저장할 위치

	// image size = 36 x 36 -> 49 blocks, 9-bin histogram per block
	float* mag = (float*)calloc(height * width, sizeof(float));
	float* dir = (float*)calloc(height * width, sizeof(float));
	float* hist = (float*)calloc(binNum, sizeof(float));
	//float* deg = (float*)calloc(height * width, sizeof(float));

	int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	Mat result(height, width, CV_8UC1);

	// compute magnitude and direction
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			conv_x = 0;
			conv_y = 0;

			//inner product
			for (yy = y - 1; yy <= y + 1; yy++) {
				for (xx = x - 1; xx <= x + 1; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						conv_x += input.at<uchar>(yy, xx) * mask_x[(yy - (y - 1)) * mask_size + xx - (x - 1)];
						conv_y += input.at<uchar>(yy, xx) * mask_y[(yy - (y - 1)) * mask_size + xx - (x - 1)];
					}
				}
			}
			//크기를 줄여줌 (필요한 연산 x)
			conv_x /= 9.0;
			conv_y /= 9.0;

			mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y);
			dir[y * width + x] = atan2(conv_y, conv_x);

			dir[y * width + x] = dir[y * width + x] * 180 / PI;

			if (dir[y * width + x] < 0)	dir[y * width + x] += 180.0;
		}
	}

	int block_idx = 0;	// block counter

	// Compute HoG
	for (y = 0; y <= height - BLOCK; y += BLOCK / 2) {
		for (x = 0; x <= width - BLOCK; x += BLOCK / 2) {

			// initialize hist array
			for (i = 0; i < binNum; i++)
				hist[i] = 0;

			for (yy = y; yy < y + BLOCK; yy++) {
				for (xx = x; xx < x + BLOCK; xx++) {
					idx = dir[yy * width + xx] / 20.0;	// Quantization
					hist[idx] += mag[yy * width + xx];
				}
			}

			// normalization
			hist_nom = 0;
			for (i = 0; i < binNum; i++)
				hist_nom += hist[i] * hist[i];

			for (i = 0; i < binNum; i++) {
				hist[i] /= sqrt(hist_nom);
				if (hist_nom == 0)
					hist[i] = 0;

				// concatenation
				refHOG_descriptor[block_idx * binNum + i] = hist[i];
			}

			block_idx++;
		}
	}

	free(mag);
	free(dir);
	//free(deg);
	free(hist);
}