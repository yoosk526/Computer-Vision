#include "main.h"

void LBP_descriptor(Mat input, Mat shape, float* LBP_hist, int numLandmarks) {
	int x, y, yy, xx, sx, sy, k, temp, i = 0;
	int mark_idx = 0;
	int val;
	float hist_nom;

	// 0번과 1번의 거리를 WIN 사이즈로 정함
	int x_d = (shape.at<float>(0) - shape.at<float>(1)) * (shape.at<float>(0) - shape.at<float>(1));
	int y_d = (shape.at<float>(0 + numLandmarks) - shape.at<float>(1 + numLandmarks)) * (shape.at<float>(0 + numLandmarks) - shape.at<float>(1 + numLandmarks));
	int WIN = int(sqrt(x_d + y_d));
	WIN /= 2;
	if (WIN % 2 == 0) {		// WIN 크기가 짝수인 경우 홀수로 만들어줌
		WIN--;
	}

	// LBP 값을 잠시 저장할 변수
	float* temp_LBP = (float*)calloc(WIN * WIN, sizeof(float));

	int width = input.cols;
	int height = input.rows;

	int start_idx = 27;		// nose, eye, mouth = 27 ~ 67

	Mat inputGray;
	cvtColor(input, inputGray, CV_BGR2GRAY);

	while (1) {
		if (start_idx == 68)
			break;
		sx = shape.at<float>(start_idx);
		sy = shape.at<float>(start_idx + numLandmarks);
		i = 0;

		for (y = sy - WIN / 2; y <= sy + WIN / 2; y++) {
			for (x = sx - WIN / 2; x <= sx + WIN / 2; x++) {
				val = 0;
				if (y - 1 >= 0 && y + 1 < height && x - 1 >= 0 && x + 1 < width) {
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1,	   x))	val += 1;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x + 1))	val += 2;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(	y, x + 1))	val += 4;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x + 1))	val += 8;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1,	   x))	val += 16;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x - 1))	val += 32;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(	y, x - 1))	val += 64;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x - 1))	val += 128;
				}
				temp_LBP[i++] = lookup[val];
			}
		}

		for (i = 0; i < WIN * WIN; i++) {
			temp = temp_LBP[i];
			LBP_hist[mark_idx * binNum + temp]++;
		}

		hist_nom = 0;
		for (i = 0; i < binNum; i++) {
			hist_nom += LBP_hist[mark_idx * binNum + i] * LBP_hist[mark_idx * binNum + i];
		}
		hist_nom = sqrt(hist_nom);

		for (i = 0; i < binNum; i++) {
			if (hist_nom != 0)
				LBP_hist[mark_idx * binNum + i] = LBP_hist[mark_idx * binNum + i] / hist_nom;
			else
				LBP_hist[mark_idx * binNum + i] = 0;
		}

		mark_idx++;
		start_idx++;
	}
	//printf("%d, done\n", mark_idx);

	free(temp_LBP);
}