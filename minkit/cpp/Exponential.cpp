#include <cmath>

extern "C" {
  /** Definition of an Exponential PDF.
   */
  static inline double shared_function( double x, double k ) {
    return std::exp(k * x);
  }

  /** Definition of an Exponential PDF.
   */
  double function( double x, double k ) {
    return shared_function(x, k);
  }

  /** Definition of the evaluation of an Exponential PDF.
   */
  void evaluate( int len, double *out, double* in, double k ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], k);
  }

  /** Normalization for an Exponential PDF.
   */
  double normalization( double k, double xmin, double xmax ) {

    return 1. / k * (std::exp(k * xmax) - std::exp(k * xmin));
  }
}
