#include "main.h"

void LBP_descriptor(Mat input, float* LBP_hist, int sx, int sy, int face_width, int face_height, int enroll) {
	int x, y, yy, xx, k, block_idx = 0;
	int val;
	float hist_nom;

	Mat inputGray;
	Mat LBP_image(face_height, face_width, CV_8UC1);
	Mat LBP_image_Resize(WIN, WIN, CV_8UC1);

	cvtColor(input, inputGray, CV_BGR2GRAY);

	for (y = sy; y < sy + face_height; y++) {
		for (x = sx; x < sx + face_width; x++) {
			val = 0;

			if (y - 1 >= sy && y + 1 < sy + face_height && x - 1 >= sx && x + 1 < sx + face_width) {
				if (input.at<uchar>(y, x) < input.at<uchar>(y - 1,		x))	val += 1;
				if (input.at<uchar>(y, x) < input.at<uchar>(y - 1,	x + 1))	val += 2;
				if (input.at<uchar>(y, x) < input.at<uchar>(	y,	x + 1))	val += 4;
				if (input.at<uchar>(y, x) < input.at<uchar>(y + 1,	x + 1))	val += 8;
				if (input.at<uchar>(y, x) < input.at<uchar>(y + 1,		x))	val += 16;
				if (input.at<uchar>(y, x) < input.at<uchar>(y + 1,	x - 1))	val += 32;
				if (input.at<uchar>(y, x) < input.at<uchar>(	y,	x - 1))	val += 64;
				if (input.at<uchar>(y, x) < input.at<uchar>(y - 1,	x - 1))	val += 128;
			}

			LBP_image.at<uchar>(y - sy, x - sx) = lookup[val];
		}
	}

	resize(LBP_image, LBP_image_Resize, Size(WIN, WIN), 1);

	// LBP histogram
	for (y = 0; y < WIN - BLK; y += BLK / 2) {
		for (x = 0; x < WIN - BLK; x += BLK / 2) {
			for (yy = 0; yy < BLK; yy++) {
				for (xx = 0; xx < BLK; xx++) {
					LBP_hist[block_idx * binNum + LBP_image_Resize.at<uchar>(yy, xx)]++;
				}
			}
			
			// normalization
			hist_nom = 0;
			for (k = 0; k < binNum; k++) {
				hist_nom += LBP_hist[block_idx * binNum + k] * LBP_hist[block_idx * binNum + k];
			}
			hist_nom = sqrt(hist_nom);

			for (k = 0; k < binNum; k++) {
				if (hist_nom != 0)
					LBP_hist[block_idx * binNum + k] = LBP_hist[block_idx * binNum + k] / hist_nom;
				else
					LBP_hist[block_idx * binNum + k] = 0;
			}

			block_idx++;
		}
	}

	if (enroll == 1)
		imwrite("../../../media/ref_LBP_image.bmp", LBP_image_Resize);		// save reference image
}