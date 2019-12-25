#include <cmath>

extern "C" {
  /** Definition of an Amoroso PDF. The normalization is not included as it explodes easily.
   */
  static inline double shared_function( double x, double a, double theta, double alpha, double beta ) {
    double d = (x - a) / theta;
    return std::pow(d, alpha * beta - 1) * std::exp(- std::pow(d, beta));
  }

  /** Definition of an Amoroso PDF.
   */
  double function( double x, double a, double theta, double alpha, double beta ) {
    return shared_function(x, a, theta, alpha, beta);
  }

  /** Definition of the evaluation of an Amoroso PDF.
   */
  void evaluate( int len, double *out, double* in, double a, double theta, double alpha, double beta ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], a, theta, alpha, beta);
  }
}
