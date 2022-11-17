// "../../media/edge_building.jpg"
#include "main.h"

void main() {

	// face recognition
	Mat ref_img = imread("../../../media/face_ref.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat tar_img = imread("../../../media/face_tar.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	/*
	int block_num = (ref_img.cols / (BLOCK / 2) - 1) * (ref_img.rows / (BLOCK / 2) - 1);		// 25 blocks

	float** ref_HoG = (float**)calloc(block_num, sizeof(float*));
	for (int i = 0; i < block_num; i++) {
		ref_HoG[i] = (float*)calloc(binNum, sizeof(float));		// 9-bin histogram
	}

	HoG_compute(ref_img, ref_HoG);
	search_Face(ref_HoG, tar_img);
	*/


	float* refHOG_descriptor = (float*)calloc(binNum * 25, sizeof(float));      // 25 = [(width / (BLOCK/2)) - 1] * [(height / (BLOCK/2)) - 1]

	ref_HoG(ref_img, refHOG_descriptor);
	searchFaces(tar_img, refHOG_descriptor);

	free(refHOG_descriptor);
}