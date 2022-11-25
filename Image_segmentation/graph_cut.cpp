#include "main.h"

void graph_cut(Mat input, Rect point) {
	Mat result;
	Mat bg, fg;

	grabCut(input, result, point, bg, fg, 8, GC_INIT_WITH_RECT);
	compare(result, GC_PR_FGD, result, CMP_EQ);

	Mat foreground(input.size(), CV_8UC3, Scalar(255, 255, 255));
	input.copyTo(foreground, result);

	imshow("Foreground", foreground);
	waitKey(0);
}