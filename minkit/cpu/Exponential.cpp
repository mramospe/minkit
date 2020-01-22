#include <cmath>

extern "C" {
  /// Function shared by "normalization" and "evaluate_binned"
  static inline double integral( double k, double xmin, double xmax ) {
    return 1. / k * (std::exp(k * xmax) - std::exp(k * xmin));
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, double k ) {
    return std::exp(k * x);
  }

  /// Function for single-value
  double function( double x, double k ) {
    return shared_function(x, k);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double* in, double k ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], k);
  }

  /// Evaluate on a binned data set
  void evaluate_binned( int len, double *out, double k, int nedges, int gap, double *edges )
  {
    for ( int i = 0; i < len; ++i ) {

      int ie = (i / gap) % nedges;

      out[i] = integral(k, edges[ie], edges[ie + 1]);
    }
  }

  /// Normalization
  double normalization( double k, double xmin, double xmax ) {
    return integral(k, xmin, xmax);
  }
}
