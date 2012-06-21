#include "viterbi.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <limits>


// Log of prob, avoid introducing NA's with 0's or negatives
double pLog(double x)
{
	// return (x <= 0.0 ? (-std::numeric_limits<double>::max()) : log(x));
	// return (x <= 0.0 ? (-1e300) : log(x));
	return log(x + 1e-300);
}

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
	int32_t *vit)
{
	// Allocate working variables
	std::vector<std::vector<double> > delta(m, std::vector<double>(n, 0.0));
	std::vector<std::vector<int> >    xsi(m, std::vector<int>(n, 0));
	std::vector<std::vector<int> >    lead;
	// double pz = -1e300;

	// Populate lead if used
	// lead_vec represents a ragged vector of vectors: [[1, 2], [7, 2, 2]]
	// as a vector: [(2), 1, 2, (3), 7, 2, 2]
	if (k > 0) {
		lead.resize(m);
		size_t at = 0;
		for (size_t i=0; i < (size_t) m; i++) {
			lead[i] = std::vector<int>(lead_vec + at + 1, lead_vec + at + 1 + lead_vec[at]);
			at += lead_vec[at] + 1;
		}

	}

	// Initialization
	for (int i = 0; i < m; i++) { // i * columns + j
		delta[i][0] = pLog(Pi[i]) + pLog(P[i * a + x[0]]);
	}

//	for (size_t i = 0; i < m; i++) {
//		std::cout << "delta[" << i << ", " << 0 << "] = " << delta[i][0] << "\n";
//	}

	// Recursion
	for (int t=1; t < n; t++) {
		for (int j=0; j < m; j++) {
			double sup=0;
			if (lead.size() > 0) {
				// Use sparsity information in lead to reduce computation
				for (size_t I=0; I < lead[j].size(); I++) {
					int i = lead[j][I];
					double term = delta[i][t-1] + pLog(Pij[i * m + j]);
					if (I == 0 || sup < term) {sup = term; xsi[j][t] = i;}
				}
			} else {
				// No sparsity info check all states
				for (int i=0; i < m; i++) {
					double term = delta[i][t-1] + pLog(Pij[i * m + j]);
					if (sup < term) {sup = term; xsi[j][t] = i;}
				}
			}
			delta[j][t] = sup + pLog(P[j * a + x[t]]);
 		}
	}

	// Termination
	double dmx=0.0;
	for (int i = 0; i < m; i++) {
		if (i == 0 || delta[i][n-1] > dmx) {
			vit[n-1] = i;
			dmx = delta[i][n-1];
		}
	}

	// Backtracking
	for (int t=n-2; t >= 0; t--) {
		vit[t] = xsi[vit[t+1]][t+1];
	}

}
