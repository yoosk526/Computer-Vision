#include "main.h"

void main() {
	int k, enroll = 1;
	float score, th = 150;
	float* ref_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));
	float* tar_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));

	VideoCapture capture(0);
	Mat frame;

	if (!capture.isOpened()) {
		printf("Couldn't open the web camera...\n");
		return;
	}

	CascadeClassifier face_cascade;
	face_cascade.load("C:/Users/yoosk/Documents/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
	vector<Rect> faces;

	while (1) {
		capture >> frame;

		face_cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		// enrollment
		if (faces.size() == 1 && enroll == 1) {
			imwrite("../../../media/ref_image.bmp", frame);		// save original image

			LBP_descriptor(frame, ref_LBP_hist, faces[0].x, faces[0].y, faces[0].width, faces[0].height, enroll);
			
			enroll = 0;
		}

		// verification
		if (faces.size() > 0 && enroll != 1) {
			for (k = 0; k < faces.size(); k++) {

				LBP_descriptor(frame, tar_LBP_hist, faces[k].x, faces[k].y, faces[k].width, faces[k].height, enroll);

				score = compute_similarity(ref_LBP_hist, tar_LBP_hist);
				printf("score : %f\n", score);

				if (score > th) {
					Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
					Point tr(faces[k].x, faces[k].y);

					rectangle(frame, lb, tr, Scalar(0, 255, 0), 2, 8, 0);		// draw green rectangle
				}
				else {
					Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
					Point tr(faces[k].x, faces[k].y);

					rectangle(frame, lb, tr, Scalar(0, 0, 255), 2, 8, 0);		// draw green rectangle
				}
			}
		}
						
		imshow("Video", frame);

		if (waitKey(30) >= 0)	
			break;
	}

	free(ref_LBP_hist);
	free(tar_LBP_hist);
}