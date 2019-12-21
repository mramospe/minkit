#include <cmath>

extern "C" {
  /** Definition of an Exponential PDF.
   */
  void Exponential( int len, double *out, double* in, double k ) {

    for ( int i = 0; i < len; ++i ) {

      double x = in[i];

      out[i] = x < 0 ? 0 : std::exp(k * x);
    }
  }

  /** Normalization for an Exponential PDF.
   */
  double normalization( double k, double xmin, double xmax ) {

    return 1. / k * (std::exp(k * xmax) - std::exp(k * xmin));
  }
}
