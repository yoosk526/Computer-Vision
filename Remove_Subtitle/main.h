#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

#define win_size 3
#define corner_box 5.0
#define ErrRange 20
#define rectRegion 5


void CornerDetection(Mat input, Mat input_gray, int* cornerMap, int* cornerNum);
void draw_rectangle(Mat input, int* cornerMap, int cornerNum, Mat mask_New);
float compute_density(Mat input, int* cornerMap, int x, int y, int width, int height);
void graph_cut_mask(Mat input, Mat maskImg, Mat mask_result);
