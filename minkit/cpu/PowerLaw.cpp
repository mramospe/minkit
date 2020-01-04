#include <cmath>

extern "C" {
  /** Definition of a PowerLaw PDF.
   */
  static inline double shared_function( double x, double c, double n ) {

    return 1. / std::pow(std::abs(x - c), n);
  }

  /** Definition of a PowerLaw PDF.
   */
  double function( double x, double c, double n ) {
    return shared_function(x, c, n);
  }

  /** Definition of the evaluation of a PowerLaw PDF.
   */
  void evaluate( int len, double *out, double* in, double c, double n ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], c, n);
  }

  /** Normalization for a PowerLaw PDF.
   */
  double normalization( double c, double n, double xmin, double xmax ) {

    bool use_log = ( std::abs(n - 1.) < 1e-5 );

    double result = 0.;

    if ( use_log )
      result = (std::log(xmax - c) - std::log(xmin - c));
    else
      result = 1. / (1. - n) * (1. / std::pow(std::abs(xmax - c), n - 1.) - 1. / std::pow(std::abs(xmin - c), n - 1.));

    return result != 0. ? result : 1e-300;
  }
}
