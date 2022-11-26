#include "main.h"

#define COSINE
//#define EUCLIDEAN

float compute_similarity(float* ref, float* tar) {
	int dim = binNum * blockNum;
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

	return exp(7 * score);
#endif

#ifdef EUCLIDEAN
	for (k = 0; k < dim; k++) {		// numBin = BIN * (numLandmarks - 17)
		score += abs(ref[k] - tar[k]);
	}

	return score;
#endif
}