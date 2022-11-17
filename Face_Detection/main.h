#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define PI 3.14159265358979323846264338327950288419716939937510
#define BLOCK 12		// width / mask_size , block size = 12 x 12
#define binNum 9
#define mask_size 3

void ref_HoG(Mat input, float* refHOG_descriptor);
void tar_HoG(float* mag, float* dir, float* tarHOG_descriptor);
void searchFaces(Mat input, float* refHOG_descriptor);
float compute_Similarity(float* ref, float* tar);