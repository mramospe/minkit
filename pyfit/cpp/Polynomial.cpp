#include <cmath>

extern "C" {
  /** Definition of a Polynomial PDF.
   */
  static inline double shared_function( double x, int n, double *p ) {
    double out = *p;
    for ( int i = 1; i < n; ++i )
      out += x * out + *(p + i);
    return out;
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

    // Right integral
    double r = (*p) / n;
    for ( int i = 1; i < n; ++i )
      r += xmax * r + (*(p + i)) / (n - i);

    // Left integral
    double l = (*p) / n;
    for ( int i = 1; i < n; ++i )
      l += xmin * l + (*(p + i)) / (n - i);

    return r * xmax - l * xmin;
  }
}
