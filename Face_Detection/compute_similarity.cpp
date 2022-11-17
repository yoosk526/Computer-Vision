#include "main.h"

#define COSINE
//#define EUCLIDEAN

float compute_Similarity(float* ref, float* tar) {
	// 25 * 9 = [(width / (BLOCK/2)) - 1] * [(height / (BLOCK/2)) - 1] * 9 bin
	int dim = 25 * 9;
	float score;

#ifdef COSINE
	int k;
	float nomi = 0, denomi;
	float refMag = 0, tarMag = 0;

	for (k = 0; k < dim; k++) {
		nomi += ref[k] * tar[k];
		refMag += ref[k] * ref[k];
		tarMag += tar[k] * tar[k];
	}
	denomi = sqrt(refMag) * sqrt(tarMag);
	score = nomi / denomi;

#endif

	return exp(20 * score);
}