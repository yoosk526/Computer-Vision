#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#define win_size 3

using namespace cv;

void CornerDetection(Mat input_color, Mat input, Mat output);