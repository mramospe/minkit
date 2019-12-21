#include <cmath>

// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001

extern "C" {
  /** Definition of a Gaussian PDF.
   */
  void Gaussian( int len, double *out, double* in, double c, double s ) {

    for ( int i = 0; i < len; ++i ) {

      double x = in[i];
      double d = x - c;
      double s2 = s * s;

      out[i] = std::exp(- d * d / ( 2. * s2 ) );
    }
  }

  /** Normalization for a Gaussian PDF.
   */
  double normalization( double c, double s, double xmin, double xmax ) {

    double sqrt2s = M_SQRT2 * s;
    double pmin   = (xmin - c) / sqrt2s;
    double pmax   = (xmax - c) / sqrt2s;
    return SQRTPI_2 * s * (std::erf(xmax) - std::erf(xmin));
  }
}
