#include "main.h"

void searchFaces(Mat input, float* refHOG_descriptor) {     // input = target image
    int x, y, xx, yy;
    int height = input.rows;
    int width = input.cols;
    int WIN = 36;
    int idx, block_idx = 0;
    float conv_x, conv_y;
    float hist_nom;
    float min = 1000000, max = -1;

    float* mag = (float*)calloc(height * width, sizeof(float));
    float* dir = (float*)calloc(height * width, sizeof(float));
    float* testMag = (float*)calloc(WIN * WIN, sizeof(float));
    float* testDir = (float*)calloc(WIN * WIN, sizeof(float));
    float* tarHOG_descriptor = (float*)calloc(binNum * 25, sizeof(float));      // 25 = [(width / (BLOCK/2)) - 1] * [(height / (BLOCK/2)) - 1]
    float* simMap = (float*)calloc(height * width, sizeof(float));

    int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    Mat result(height, width, CV_8UC1);
    Mat faceResult(height, width, CV_8UC3);

    // compute magnitude and direction
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            conv_x = 0;
            conv_y = 0;

            for (yy = y - 1; yy <= y + 1; yy++) {
                for (xx = x - 1; xx <= x + 1; xx++)
                {
                    if (yy >= 0 && xx >= 0 && yy < height && xx < width)
                    {
                        conv_y += input.at<uchar>(yy, xx) * mask_y[(yy - (y - 1)) * mask_size + xx - (x - 1)];
                        conv_x += input.at<uchar>(yy, xx) * mask_x[(yy - (y - 1)) * mask_size + xx - (x - 1)];
                    }
                }
            }

            conv_x /= 9.0;
            conv_y /= 9.0;

            mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y);
            dir[y * width + x] = atan2(conv_y, conv_x);

            dir[y * width + x] = dir[y * width + x] * 180.0 / PI;
            if (dir[y * width + x] < 0) dir[y * width + x] += 180.0;

            faceResult.at<Vec3b>(y, x)[0] = input.at<uchar>(y, x);
            faceResult.at<Vec3b>(y, x)[1] = input.at<uchar>(y, x);
            faceResult.at<Vec3b>(y, x)[2] = input.at<uchar>(y, x);
        }
    }

    // searchihg faces
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            // make a block
            idx = 0;

            // criteria point is center 
            for (yy = y - WIN / 2; yy < y + WIN / 2; yy++) {
                for (xx = x - WIN / 2; xx < x + WIN / 2; xx++) {
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        testMag[idx] = mag[yy * width + xx];
                        testDir[idx] = dir[yy * width + xx];
                        idx++;
                    }
                }
            }

            //compute tarHOG
            tar_HoG(testMag, testDir, tarHOG_descriptor);

            //comput similarity
            simMap[y * width + x] = compute_Similarity(refHOG_descriptor, tarHOG_descriptor);

            if (min > simMap[y * width + x])    min = simMap[y * width + x];
            if (max < simMap[y * width + x])    max = simMap[y * width + x];
        }
    }

    // draw a score map
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            // normalization
            // simMap[y * width + x] = (simMap[y * width + x] - min) / (max - min);
            result.at<uchar>(y, x) = 255 * (simMap[y * width + x] - min) / (max - min);
        }
    }

    int* no_overlap = (int*)calloc(height * width, sizeof(int));
    int no_draw = 0;

    // draw face regions
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            for (yy = y - WIN / 2; yy < y + WIN / 2; yy++) {
                for (xx = x - WIN / 2; xx < x + WIN / 2; xx++) {
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        if (no_overlap[yy * width + xx] == 1)
                            no_draw = 1;
                    }
                }
            }
            if (sqrt(simMap[y * width + x]) > 0.4 * sqrt(max)) {     // 0.89
                // printf("x : %d, y : %d \t %f\n", x, y, simMap[y * width + x]);

                faceResult.at<Vec3b>(y, x)[0] = 255;
                faceResult.at<Vec3b>(y, x)[1] = 0;
                faceResult.at<Vec3b>(y, x)[2] = 255;

                if (no_draw == 0) {
                    rectangle(faceResult, Point(x - WIN / 2, y - WIN / 2), Point(x + WIN / 2, y + WIN / 2), Scalar(255, 0, 255), 1, 8, 0);
                    no_overlap[y * width + x] = 1;
                }
                no_draw = 0;
            }
        }
    }

    imwrite("../../media/similarityMap.bmp", result);
    imwrite("../../media/detectionResult.bmp", faceResult);
    imshow("similarityMap", result);
    imshow("detectionResult.bmp", faceResult);
    waitKey(0);
    destroyAllWindows();

    free(mag);
    free(dir);
    free(testMag);
    free(testDir);
    free(tarHOG_descriptor);
    free(simMap);
    free(no_overlap);
}