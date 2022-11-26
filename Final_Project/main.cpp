#include "main.h"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "ldmarkmodel.h"



// login delay�� �� ������ ��ȯ�Ͽ� ���� ����ϱ� ���� ���ڿ��� �ٲٴ� �Լ�
void timeout(int i, char * text) {	
	if (i == 3)
		sprintf(text, "3");
	else if (i == 2)
		sprintf(text, "2");
	else if (i == 1)
		sprintf(text, "1");
}

// Mouse callback �Լ����� ���Ǵ� ���� ���� ����
Point pt;		
Mat maskImg;	

void on_mouse(int event, int x, int y, int flags, void* userdata);

void brush_size_change(int size, void* userdata);

void superpixel_num_change(int sup, void* userdata);

void compactness_change(int com, void* userdata);

int main()
{
	/*=================================     Login     ==================================*/
	int login_flag = 1;
	int login_delay = 0;
	char put_time[2];		// ���� ����� string
	Mat ready = imread("../../../media/ready.png");
	Mat login = imread("../../../media/login.png");
	resize(ready, ready, Size(640, 480));
	resize(login, login, Size(640, 480));

	int enroll = 1;				// 1 : ���� ����ؾ� ��
	float score, th = 120;		// ���� ���� ����

	// binNum : 59 uniform LBP, blockNum = 41 (eyes, nose, mouth)
	float* ref_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));		
	float* tar_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));

	// �н� ������ �ҷ�����
	ldmarkmodel modelt;
	std::string modelFilePath = "roboman-landmark-model.bin";
	while (!load_ldmarkmodel(modelFilePath, modelt)) {
		std::cout << "Can not load landmark model" << std::endl;
		std::cin >> modelFilePath;
	}

	cv::VideoCapture mCamera(0);
	if (!mCamera.isOpened()) {
		cout << "Camera opening failed..." << endl;
		system("pause");
		return 0;
	}

	Mat Image;
	Mat current_shape;
	
	// ���� ������ Ȯ��
	double fps = mCamera.get(CV_CAP_PROP_FPS);
	printf("Camera FPS : %f\n", fps);

	// �� ����ϱ��� �غ� �̹��� ���
	imshow("Ready", ready);
	waitKey(0);
	destroyWindow("Ready");

	for (;;) {
		mCamera >> Image;
		modelt.track(Image, current_shape);
		Vec3d eav;
		modelt.EstimateHeadPose(current_shape, eav);
		modelt.drawPose(Image, current_shape, 50);

		int numLandmarks = current_shape.cols / 2;

		if (numLandmarks != 0 && enroll == 1) {		// Landmark�� ����ǰ�, enroll = 1 �̸� reference�� LBP ����
			LBP_descriptor(Image, current_shape, ref_LBP_hist, numLandmarks);

			enroll = 0;		// ���� reference LBP�� �ٽ� ���� �ʿ䰡 �����Ƿ� flag ����
		}

		if (numLandmarks != 0 && enroll == 0) {
			LBP_descriptor(Image, current_shape, tar_LBP_hist, numLandmarks);	// target�� LBP ����
			score = compute_similarity(ref_LBP_hist, tar_LBP_hist);				// ���絵 ���

			printf("score : %f, login delay : %d\n", score, login_delay);

			if (score > th) {
				login_delay++;
				timeout(3 - login_delay / 30, put_time);	// ���� �� �� ���Ҵ��� ����ϱ� ���� ���ڿ��� �ٲٴ� ����

				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);		// �� Landmark�� �ʷϻ� �� �׸�
				}
				rectangle(Image, Rect(Image.cols * 0.85, Image.rows * 0.16, 70, 70), Scalar(125, 255, 125), -1);	// Green rectangle
				putText(Image, put_time, Point(Image.cols * 0.88, Image.rows * 0.27), 6, 1.7, (0, 0, 0), 3);		// �� �� ���Ҵ��� ���
			}
			// �� ������ ������ ���
			else {
				login_delay = 0;	// �ٽ� �ʱ� delay�� �ʱ�ȭ
				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);		// �� Landmark�� ������ �� �׸�
				}
				rectangle(Image, Rect(Image.cols * 0.85, Image.rows * 0.16, 70, 70), Scalar(0, 0, 255), -1);		// Red rectangle
			}
		}

		imshow("Camera", Image);

		if (waitKey(1) == 27) {			// ESC ������ ����
			login_flag = 0;				// login�� ���������Ƿ� ���� grabcut, SLIC �Ұ���
			mCamera.release();
			destroyAllWindows();
			break;
		}
		if (login_delay == 90) {		// 3�ʰ� ����� ���� Ȯ�εǸ� �α���
			imshow("Camera", login);
			waitKey(2000);
			mCamera.release();
			destroyAllWindows();
			break;
		}
	}

	// ���� �Ҵ� ����
	free(ref_LBP_hist);
	free(tar_LBP_hist);

	system("pause");


	/*=================================     Graph cut & SLIC     ==================================*/
	int height, width;
	int process_flag = 0;
	int mask_flag = 0;			// mask �̹����� ������� ������ 1
	int st_size = 0;			// Trackbar���� �ʱ� ��ġ
	int brush_size = 20;		// �귯�� ũ�� ������ ���� ����
	int show_result_flag = 0;

	int m_spcount = 100;		// number of superpixels
	int m_compactness = 20;		// compactness factor (1-40)
	int slic_flag = 0;
	int init_set = 1;
	Mat slic;

	if (login_flag == 1) {
		VideoCapture capture(0);
		
		while (1) {
			capture >> Image;
			if (process_flag == 0)
				imshow("Camera", Image);

			int keycode = waitKey(1);

			// Mask �̹��� �����
			// �� frame���� Ű���� �Է��� ���� �� �ֱ� ������ ���Ҷ� ����ũ �̹��� ���� ����
			if (keycode == 's' || keycode == 'S') {
				imwrite("./images/original.jpg", Image);		// Graph cut�ϰ� ���� �̹��� ����
				maskImg = imread("./images/original.jpg");		// ������ �̹��� �ҷ�����

				namedWindow("Making Mask Image");
				createTrackbar("brush size", "Making Mask Image", &st_size, 40, brush_size_change, (void*)&brush_size);		// Trackbar ����, ����� �����ͷ� brush_size�� �Ѱ���
				setTrackbarPos("brush size", "Making Mask Image", 20);						// Trackbar�� ó�� ����
				setMouseCallback("Making Mask Image", on_mouse, (void*)&brush_size);		// Mouse �ݹ� �Լ� ȣ��

				imshow("Making Mask Image", maskImg);
				waitKey(0);

				destroyWindow("Making Mask Image");
				destroyWindow("Camera");
				mask_flag = 1;
				process_flag = 1;
				show_result_flag = 1;
			}
		
			// Mask �̹����� grabcut ����
			if (mask_flag == 1 && show_result_flag == 1) {
				printf("\n=============== Graph cut by mask ===============\n");
				printf("\n=============== SLIC segmentation ===============\n");
				maskImg = imread("./images/maskImg.bmp");
				height = Image.rows;
				width = Image.cols;
				Mat final_result = Mat::zeros(height, width, CV_8UC3);

				graph_cut_mask(Image, maskImg, final_result);

				slic = Image;

				show_result_flag = 0;
				slic_flag = 1;
			}

			if (slic_flag == 1) {
				namedWindow("SLIC segmentation");
				if (init_set == 1) {
					// Superpixel�� ������ compactness ���� �����ϱ� ���� Trackbar
					// init_set ó�� flag�� ���� ������ Trackbar �� ���� �������� �ʰ� ��� ������
					createTrackbar("# of Superpixel", "SLIC segmentation", 0, 400, superpixel_num_change, (void*)&m_spcount);
					createTrackbar("Compactness", "SLIC segmentation", 0, 40, compactness_change, (void*)&m_compactness);
					setTrackbarPos("# of Superpixel", "SLIC segmentation", 100);
					setTrackbarPos("Compactness", "SLIC segmentation", 20);
					init_set = 0;
				}				
				SLICsegmentation(slic, m_spcount, m_compactness);		// ���� ���� ���ؼ� SLIC ����
			}

			// 'c' �� 'C'�� ������ ��ķ ������ ������ ��� â �ݰ�, flag �ʱ�ȭ
			if (keycode == 'c' || keycode == 'C') {		
				destroyWindow("graph_cut_result");
				destroyWindow("SLIC segmentation");
				init_set = 1;
				slic_flag = 0;
				process_flag = 0;
				show_result_flag = 0;
			}
			
			if (keycode == 27) {		// ESC�� ������ ����
				capture.release();
				destroyAllWindows();
				break;
			}
		}
	}
	else
		return 0;
	
	return 0;
}


void on_mouse(int event, int x, int y, int flags, void* userdata) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
		pt = Point(x, y);
		printf("Left button down : (%d, %d)\n", x, y);
		break;
	case EVENT_LBUTTONUP:
		printf("Left button up : (%d, %d)\n", x, y);
		break;
	case EVENT_RBUTTONDOWN:
		pt = Point(x, y);
		printf("Right button down : (%d, %d)\n", x, y);
		break;
	case EVENT_RBUTTONUP:
		printf("Right button up : (%d, %d)\n", x, y);
		break;
	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON) {
			line(maskImg, pt, Point(x, y), Scalar(0, 0, 255), *(int*)userdata);       // ���� ���콺 : Red
			imshow("Making Mask Image", maskImg);
			pt = Point(x, y);
		}
		else if (flags & EVENT_FLAG_RBUTTON) {
			line(maskImg, pt, Point(x, y), Scalar(255, 0, 0), *(int*)userdata);       // ������ ���콺 : Blue
			imshow("Making Mask Image", maskImg);
			pt = Point(x, y);
		}
		break;
	case EVENT_MBUTTONDOWN:
		printf("\n=============== Save Mask Image ===============\n");
		imwrite("./images/maskImg.bmp", maskImg);
		break;
	default:
		break;
	}
}

void brush_size_change(int size, void* userdata) {
	//printf("brush size : %d\n", size);
	*(int*)userdata = size;
}

void superpixel_num_change(int sup, void* userdata) {
	*(int*)userdata = sup;
}

void compactness_change(int com, void* userdata) {
	*(int*)userdata = com;
}
