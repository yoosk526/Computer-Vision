#include "main.h"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "ldmarkmodel.h"



// login delay를 초 단위로 변환하여 영상에 출력하기 위해 문자열로 바꾸는 함수
void timeout(int i, char * text) {	
	if (i == 3)
		sprintf(text, "3");
	else if (i == 2)
		sprintf(text, "2");
	else if (i == 1)
		sprintf(text, "1");
}

// Mouse callback 함수에서 사용되는 전역 변수 선언
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
	char put_time[2];		// 영상에 출력할 string
	Mat ready = imread("../../../media/ready.png");
	Mat login = imread("../../../media/login.png");
	resize(ready, ready, Size(640, 480));
	resize(login, login, Size(640, 480));

	int enroll = 1;				// 1 : 현재 등록해야 함
	float score, th = 120;		// 본인 인증 기준

	// binNum : 59 uniform LBP, blockNum = 41 (eyes, nose, mouth)
	float* ref_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));		
	float* tar_LBP_hist = (float*)calloc(binNum * blockNum, sizeof(float));

	// 학습 데이터 불러오기
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
	
	// 영상 프레임 확인
	double fps = mCamera.get(CV_CAP_PROP_FPS);
	printf("Camera FPS : %f\n", fps);

	// 얼굴 등록하기전 준비 이미지 출력
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

		if (numLandmarks != 0 && enroll == 1) {		// Landmark가 검출되고, enroll = 1 이면 reference의 LBP 구함
			LBP_descriptor(Image, current_shape, ref_LBP_hist, numLandmarks);

			enroll = 0;		// 이제 reference LBP는 다시 구할 필요가 없으므로 flag 변경
		}

		if (numLandmarks != 0 && enroll == 0) {
			LBP_descriptor(Image, current_shape, tar_LBP_hist, numLandmarks);	// target의 LBP 구함
			score = compute_similarity(ref_LBP_hist, tar_LBP_hist);				// 유사도 계산

			printf("score : %f, login delay : %d\n", score, login_delay);

			if (score > th) {
				login_delay++;
				timeout(3 - login_delay / 30, put_time);	// 영상에 몇 초 남았는지 출력하기 위해 문자열로 바꾸는 과정

				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);		// 각 Landmark에 초록색 원 그림
				}
				rectangle(Image, Rect(Image.cols * 0.85, Image.rows * 0.16, 70, 70), Scalar(125, 255, 125), -1);	// Green rectangle
				putText(Image, put_time, Point(Image.cols * 0.88, Image.rows * 0.27), 6, 1.7, (0, 0, 0), 3);		// 몇 초 남았는지 출력
			}
			// 얼굴 인증에 실패한 경우
			else {
				login_delay = 0;	// 다시 초기 delay로 초기화
				for (int j = 0; j < numLandmarks; j++) {
					int x = current_shape.at<float>(j);
					int y = current_shape.at<float>(j + numLandmarks);
					circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);		// 각 Landmark에 빨간색 원 그림
				}
				rectangle(Image, Rect(Image.cols * 0.85, Image.rows * 0.16, 70, 70), Scalar(0, 0, 255), -1);		// Red rectangle
			}
		}

		imshow("Camera", Image);

		if (waitKey(1) == 27) {			// ESC 눌리면 종료
			login_flag = 0;				// login에 실패했으므로 밑의 grabcut, SLIC 불가능
			mCamera.release();
			destroyAllWindows();
			break;
		}
		if (login_delay == 90) {		// 3초간 사용자 임이 확인되면 로그인
			imshow("Camera", login);
			waitKey(2000);
			mCamera.release();
			destroyAllWindows();
			break;
		}
	}

	// 동적 할당 해제
	free(ref_LBP_hist);
	free(tar_LBP_hist);

	system("pause");


	/*=================================     Graph cut & SLIC     ==================================*/
	int height, width;
	int process_flag = 0;
	int mask_flag = 0;			// mask 이미지가 만들어져 있으면 1
	int st_size = 0;			// Trackbar에서 초기 위치
	int brush_size = 20;		// 브러쉬 크기 변경을 위한 변수
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

			// Mask 이미지 만들기
			// 매 frame마다 키보드 입력을 받을 수 있기 때문에 원할때 마스크 이미지 변경 가능
			if (keycode == 's' || keycode == 'S') {
				imwrite("./images/original.jpg", Image);		// Graph cut하고 싶은 이미지 저장
				maskImg = imread("./images/original.jpg");		// 저장한 이미지 불러오기

				namedWindow("Making Mask Image");
				createTrackbar("brush size", "Making Mask Image", &st_size, 40, brush_size_change, (void*)&brush_size);		// Trackbar 생성, 사용자 데이터로 brush_size를 넘겨줌
				setTrackbarPos("brush size", "Making Mask Image", 20);						// Trackbar의 처음 상태
				setMouseCallback("Making Mask Image", on_mouse, (void*)&brush_size);		// Mouse 콜백 함수 호출

				imshow("Making Mask Image", maskImg);
				waitKey(0);

				destroyWindow("Making Mask Image");
				destroyWindow("Camera");
				mask_flag = 1;
				process_flag = 1;
				show_result_flag = 1;
			}
		
			// Mask 이미지로 grabcut 진행
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
					// Superpixel의 개수와 compactness 값을 조절하기 위한 Trackbar
					// init_set 처럼 flag를 주지 않으면 Trackbar 의 값이 움직이지 않고 계속 고정됨
					createTrackbar("# of Superpixel", "SLIC segmentation", 0, 400, superpixel_num_change, (void*)&m_spcount);
					createTrackbar("Compactness", "SLIC segmentation", 0, 40, compactness_change, (void*)&m_compactness);
					setTrackbarPos("# of Superpixel", "SLIC segmentation", 100);
					setTrackbarPos("Compactness", "SLIC segmentation", 20);
					init_set = 0;
				}				
				SLICsegmentation(slic, m_spcount, m_compactness);		// 현재 영상에 대해서 SLIC 진행
			}

			// 'c' 나 'C'를 누르면 웹캠 영상을 제외한 모든 창 닫고, flag 초기화
			if (keycode == 'c' || keycode == 'C') {		
				destroyWindow("graph_cut_result");
				destroyWindow("SLIC segmentation");
				init_set = 1;
				slic_flag = 0;
				process_flag = 0;
				show_result_flag = 0;
			}
			
			if (keycode == 27) {		// ESC를 누르면 종료
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
			line(maskImg, pt, Point(x, y), Scalar(0, 0, 255), *(int*)userdata);       // 왼쪽 마우스 : Red
			imshow("Making Mask Image", maskImg);
			pt = Point(x, y);
		}
		else if (flags & EVENT_FLAG_RBUTTON) {
			line(maskImg, pt, Point(x, y), Scalar(255, 0, 0), *(int*)userdata);       // 오른쪽 마우스 : Blue
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
