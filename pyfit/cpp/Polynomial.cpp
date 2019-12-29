#include <cmath>

extern "C" {
  /** Definition of a Polynomial PDF.
   */
  static inline double shared_function( double x, int n, double *p ) {

    if ( n == 0 )
      return 1.;

    double out = p[n - 1];
    for ( int i = 1; i < n; ++i )
      out += x * out + p[n - i - 1];

    return out * x + 1.;
  }

  /** Definition of a Polynomial PDF.
   */
  double function( double x, int n, double *p ) {
    return shared_function(x, n, p);
  }

  /** Definition of the evaluation of a Polynomial PDF.
   */
  void evaluate( int len, double *out, double *in, int n, double *p ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], n, p);
  }

  /** Normalization for a Polynomial PDF.
   */
  double normalization( int n, double *p, double xmin, double xmax ) {

    if ( n == 0 )
      return xmax - xmin;

    // Right integral
    double r = p[n - 1] / (n + 1);
    for ( int i = 1; i < n; ++i )
      r += xmax * r + p[n - i - 1] / (n - i);

    // Left integral
    double l = p[n - 1] / (n + 1);
    for ( int i = 1; i < n; ++i )
      l += xmin * l + p[n - i - 1] / (n - i);

    return (1. + r) * xmax - (1. + l) * xmin;
  }
}
