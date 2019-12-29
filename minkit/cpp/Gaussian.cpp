#include <cmath>

// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001

extern "C" {
  /** Definition of a Gaussian PDF.
   */
  static inline double shared_function( double x, double c, double s ) {

    double d = x - c;
    double s2 = s * s;

    return std::exp(- d * d / ( 2. * s2 ) );
  }

  /** Definition of a Gaussian PDF.
   */
  double function( double x, double c, double s ) {
    return shared_function(x, c, s);
  }

  /** Definition of the evaluation of a Gaussian PDF.
   */
  void evaluate( int len, double *out, double* in, double c, double s ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], c, s);
  }

  /** Normalization for a Gaussian PDF.
   */
  double normalization( double c, double s, double xmin, double xmax ) {

    double sqrt2s = M_SQRT2 * s;
    double pmin   = (xmin - c) / sqrt2s;
    double pmax   = (xmax - c) / sqrt2s;

    return SQRTPI_2 * s * (std::erf(pmax) - std::erf(pmin));
  }
}
