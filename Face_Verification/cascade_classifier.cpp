#include "main.h"

void cascade_classifier(Mat input) {
	int i;
	CascadeClassifier face_cascade;
	CascadeClassifier eye_cascade;

	face_cascade.load("C:/Users/yoosk/Documents/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
	eye_cascade.load("C:/Users/yoosk/Documents/opencv/sources/data/haarcascades/haarcascade_eye.xml");


	// Mat img = imread("../../../media/face_tar.bmp", 1);	// image read (color)
	
	vector<Rect> faces;
	face_cascade.detectMultiScale(input, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	for (i = 0; i < faces.size(); i++)
	{
		Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point tr(faces[i].x, faces[i].y);
		rectangle(input, lb, tr, Scalar(0, 255, 0), 2, 8, 0);
	}
}