#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define ErrRange 10

Rect find_region(Mat input);
void graph_cut(Mat input, Rect point);
void graph_cut_mask(Mat input, Mat maskImg);
