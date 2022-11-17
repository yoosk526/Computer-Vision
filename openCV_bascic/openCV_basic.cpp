#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void main() {
	Mat imgColor = imread("../../../media/sunrising.jpg", CV_LOAD_IMAGE_COLOR);
	Mat imgGray = imread("../../../media/sunrising.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat result_x;
	Mat result_y;
	Mat result;

	Sobel(imgGray, result_x, -1, 1, 0);
	Sobel(imgGray, result_y, -1, 0, 1);
	result = result_x + result_y;

	imshow("color", imgColor);
	imshow("gray", imgGray);
	imshow("x", result_x);
	imshow("y", result_y);
	imshow("result", result);

	imwrite("../../../media/sunrising_grayscale.jpg", imgGray);

	waitKey(0);
	destroyAllWindows();
}