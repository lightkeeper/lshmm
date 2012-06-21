#include <stdint.h>

extern "C"
{
	// Calculate viterbi sequence given the hmm described by: Pi, Pij, P
	void viterbi(int32_t n,
		     const int32_t *x,
		     int32_t m,
		     int32_t a,
		     const double *Pi,
		     const double *Pij,
		     const double *P,
		     int32_t k,
		     const int32_t *lead_vec,
		     int32_t *vit);
}


// Log of prob, avoid introducing NA's with 0's or negatives
double pLog(double x);

