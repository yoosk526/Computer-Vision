#include "main.h"

LARGE_INTEGER freq, start, stop;
double diff, avg;

void main() {

	VideoCapture capture(0);
	Mat frame, frame_gray;

	char filename[200];
	int frameNo = 1;

	if (!capture.isOpened()) {
		printf("Couldn't open the web camera...\n");
		return;
	}

	while (1) {

		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&start);

		capture >> frame;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);		// Harris corner detector는 gray scale 이미지로 
		Mat cornerMap(frame.rows, frame.cols, CV_8UC3);		

		CornerDetection(frame, frame_gray, cornerMap);
		
		QueryPerformanceCounter(&stop);
		diff = (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart;
		avg += (1.0 / diff);

		//printf("\nframe : %f\n", 1.0 / diff);				// frame per sec = 1.0 / time
		printf("\nAverage frame : %f\n", avg / frameNo);		
		
		imshow("Corner Detection", cornerMap);
		
		//sprintf(filename, "./media/image%04d.bmp", frameNo);
		//imwrite(filename, frame);

		if (waitKey(30) >= 0) break;

		frameNo++;
	}
}