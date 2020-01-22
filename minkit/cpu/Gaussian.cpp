#include <cmath>

// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001

extern "C" {

  /// Function shared by "normalization" and "evaluate_binned"
  static inline double integral( double c, double s, double xmin, double xmax ) {

    double sqrt2s = M_SQRT2 * s;
    double pmin   = (xmin - c) / sqrt2s;
    double pmax   = (xmax - c) / sqrt2s;

    return SQRTPI_2 * s * (std::erf(pmax) - std::erf(pmin));
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, double c, double s ) {

    double d = (x - c) / s;

    return std::exp(-0.5 * d * d );
  }

  /// Function for single-value
  double function( double x, double c, double s ) {
    return shared_function(x, c, s);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double* in, double c, double s ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], c, s);
  }

  /// Evaluate on a binned data set
  void evaluate_binned( int len, double *out, double c, double s, int nedges, int gap, double *edges ) {

    for ( int i = 0; i < len; ++i ) {

      int ie = (i / gap) % nedges;

      out[i] = integral(c, s, edges[ie], edges[ie + 1]);
    }
  }

  /// Normalization
  double normalization( double c, double s, double xmin, double xmax ) {
    return integral(c, s, xmin, xmax);
  }
}
