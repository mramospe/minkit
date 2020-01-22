#include <cmath>

extern "C" {
  /// Function shared by "normalization" and "evaluate_binned"
  static inline double integral( int n, double *p, double xmin, double xmax ) {
    if ( n == 0 )
      return xmax - xmin;

    // Right integral
    double r = xmax * p[n - 1] / (n + 1);
    for ( int i = 1; i < n; ++i )
      r = xmax * (r + p[n - i - 1] / (n + 1 - i));

    // Left integral
    double l = xmin * p[n - 1] / (n + 1);
    for ( int i = 1; i < n; ++i )
      l = xmin * (l + p[n - i - 1] / (n + 1 - i));

    return (1. + r) * xmax - (1. + l) * xmin;
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, int n, double *p ) {

    if ( n == 0 )
      return 1.;

    double out = x * p[n - 1];
    for ( int i = 1; i < n; ++i )
      out = x * (out + p[n - i - 1]);

    return out + 1.;
  }

  /// Function for single-value
  double function( double x, int n, double *p ) {
    return shared_function(x, n, p);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double *in, int n, double *p ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], n, p);
  }

  /// Evaluate on a binned data set
  void evaluate_binned( int len, double *out, int n, double *p, int nedges, int gap, double *edges )
  {

    for ( int i = 0; i < len; ++i ) {

      int ie = (i / gap) % nedges;

      out[i] = integral(n, p, edges[ie], edges[ie + 1]);
    }
  }

  /// Normalization
  double normalization( int n, double *p, double xmin, double xmax ) {
    return integral(n, p, xmin, xmax);
  }
}
