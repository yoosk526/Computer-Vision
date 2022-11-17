#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

void main() {
	VideoCapture capture(0);
	Mat frame;

	char filename[200];
	int frameNo = 0;

	if (!capture.isOpened()) {
		printf("Couldn't open the web camera...\n");
		return;
	}

	while (1) {
		capture >> frame;
		imshow("Video", frame);
		sprintf(filename, "./media/image%04d.bmp", frameNo);
		imwrite(filename, frame);
		frameNo++;
		if (waitKey(30) >= 0) break;
	}
}