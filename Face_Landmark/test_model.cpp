#include <vector>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "ldmarkmodel.h"

#define binNum		59		// uniform LBP
#define blockNum	41		// eye, nose, mouth : 27 ~ 67

const char lookup[256] = {
0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

using namespace std;
using namespace cv;

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
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x))	val += 1;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x + 1))	val += 2;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y, x + 1))	val += 4;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x + 1))	val += 8;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x))	val += 16;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x - 1))	val += 32;
					if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y, x - 1))	val += 64;
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

#define COSINE
//#define EUCLIDEAN

float compute_similarity(float* ref, float* tar) {
	int dim = binNum * blockNum;
	float score;

#ifdef COSINE
	int k;
	float nomi = 0, denomi;
	float refMag = 0, tarMag = 0;

	for (k = 0; k < dim; k++) {
		nomi += ref[k] * tar[k];
		refMag += ref[k] * ref[k];
		tarMag += tar[k] * tar[k];
	}
	denomi = sqrt(refMag) * sqrt(tarMag);
	score = nomi / denomi;

#endif

	return exp(7 * score);
}

int main()
{
	int enroll = 1;
	float score, th = 120;
	float* ref_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));
	float* tar_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));

	ldmarkmodel modelt;
	std::string modelFilePath = "roboman-landmark-model.bin";
	while (!load_ldmarkmodel(modelFilePath, modelt)) {
		std::cout << "Can not load landmark model" << std::endl;
		std::cin >> modelFilePath;
	}

	cv::VideoCapture mCamera(0);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}

	Mat Image;              // 한 프레임을 받을 변수
	Mat current_shape;

	for (;;) {
		mCamera >> Image;
		modelt.track(Image, current_shape);
		Vec3d eav;
		modelt.EstimateHeadPose(current_shape, eav);
		modelt.drawPose(Image, current_shape, 50);

		int numLandmarks = current_shape.cols / 2;

		//printf("%d\n", numLandmarks);
		if (numLandmarks != 0 && enroll == 1) {
			LBP_descriptor(Image, current_shape, ref_LBP_hist, numLandmarks);

			enroll = 0;
		}

		if (numLandmarks != 0 && enroll == 0) {
			LBP_descriptor(Image, current_shape, tar_LBP_hist, numLandmarks);

			score = compute_similarity(ref_LBP_hist, tar_LBP_hist);

			printf("score : %f\n", score);
			if (score > th) {
				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					std::stringstream ss;
					ss << j;
					//putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
				}
			}
			else {
				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					std::stringstream ss;
					ss << j;
					//putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
				}
			}
		}

		imshow("Camera", Image);
		if (27 == waitKey(5)) {
			mCamera.release();
			destroyAllWindows();
			break;
		}
	}

	free(ref_LBP_hist);
	free(tar_LBP_hist);

	system("pause");
	return 0;
}