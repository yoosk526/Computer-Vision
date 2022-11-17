#include "main.h"

void tar_HoG(float* mag, float* dir, float* tarHOG_descriptor) {

    int x, y, xx, yy, i;
    int height = 36, width = 36;
    int idx, block_idx = 0;
    float hist_nom;
    float* hist = (float*)calloc(binNum, sizeof(float));


    //compute HOG
    for (y = 0; y <= height - BLOCK; y += BLOCK / 2) {
        for (x = 0; x <= width - BLOCK; x += BLOCK / 2) {

            for (i = 0; i < binNum; i++)
                hist[i] = 0; //ÃÊ±âÈ­


            for (yy = y; yy < y + BLOCK; yy++) {
                for (xx = x; xx < x + BLOCK; xx++) {
                    idx = dir[yy * width + xx] / 20.0;
                    hist[idx] += mag[yy * width + xx];
                }
            }

            //normalization
            hist_nom = 0;
            for (i = 0; i < binNum; i++)
                hist_nom += hist[i] * hist[i];

            for (i = 0; i < binNum; i++) {
                if (hist_nom == 0)
                    hist[i] = 0;
                else
                    hist[i] /= sqrt(hist_nom);

                //concatenation
                tarHOG_descriptor[block_idx * binNum + i] = hist[i];
            }

            block_idx++;
        }

    }

    free(hist);
}
