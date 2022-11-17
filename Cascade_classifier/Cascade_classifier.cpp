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

void main() {
	int y;
	CascadeClassifier face_cascade;
	CascadeClassifier eye_cascade;

	face_cascade.load("C:/Users/yoosk/Documents/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
	eye_cascade.load("C:/Users/yoosk/Documents/opencv/sources/data/haarcascades/haarcascade_eye.xml");

	Mat img = imread("../../../media/woman.jpg", 1); // image read (color)

	vector<Rect> faces;
	vector<Rect> eyes;

	face_cascade.detectMultiScale(img, faces, 1.05, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	eye_cascade.detectMultiScale(img, eyes, 1.02, 6, 0 | CV_HAAR_SCALE_IMAGE, Size(28, 28), Size(30, 30));

	printf("%d\n", faces.size());

	for (y = 0; y < faces.size(); y++)
	{
		Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
		Point tr(faces[y].x, faces[y].y);
		rectangle(img, lb, tr, Scalar(0, 255, 0), 2, 8, 0);
	}

	printf("%d\n", eyes.size());

	for (y = 0; y < eyes.size(); y++)
	{
		Point lb(eyes[y].x + eyes[y].width, eyes[y].y + eyes[y].height);
		Point tr(eyes[y].x, eyes[y].y);
		printf("%d\n", eyes[y].width);
		rectangle(img, lb, tr, Scalar(255, 255, 0), 2, 8, 0);
	}

	imshow("Face", img);
	waitKey(-1);
}