#include <algorithm>
#include <cmath>

extern "C" {
  /** Definition of a Chebyshev polynomial PDF.
   */
  static inline double shared_function( double x, int n, double *p ) {

    if ( n == 0 )
      return 1.;
    else if ( n == 1 )
      return 1. + p[0] * x;
    else {

      double Tipp = 1.;
      double Tip = x;
      double res = p[1] * Tip + p[0] * Tipp;

      for ( int i = 2; i < n; ++i ) {

	double Ti = 2. * x * Tip - Tipp;

	res += p[i] * Ti;

	std::swap(Tip, Tipp);

	Tip = Ti;
      }

      return res;
    }
  }

  /** Definition of a Chebyshev polynomial PDF.
   */
  double function( double x, int n, double *p ) {
    return shared_function(x, n, p);
  }

  /** Definition of the evaluation of a Chebyshev polynomial PDF.
   */
  void evaluate( int len, double *out, double *in, int n, double *p ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], n, p);
  }

  /** Normalization for a Chebyshev polynomial PDF.
   */
  double normalization( int n, double *p, double xmin, double xmax ) {

    if ( n == 1 )
      return p[0] * (xmax - xmin);
    else if ( n == 2 )
      return p[0] * (xmax - xmin) + 0.5 * p[1] * (xmax * xmax - xmin * xmin);
    else {

      double Tipp_r = 1.;
      double Tip_r = xmax;
      double Tipp_l = 1.;
      double Tip_l = xmin;
      double res = p[0] * (xmax - xmin) + 0.5 * p[1] * (xmax * xmax - xmin * xmin);

      for ( int i = 2; i < n; ++i ) {

	double Ti_r = 2. * xmax * Tip_r - Tipp_r;
	double Tin_r = 2. * xmax * Ti_r - Tip_r;

	double Ti_l = 2. * xmin * Tip_l - Tipp_l;
	double Tin_l = 2. * xmin * Ti_l - Tip_l;

	res += 0.5 * p[i] * ((Tin_r - Tin_l) / (i + 1) - (Tip_r - Tip_l) / (i - 1));

	std::swap(Tip_r, Tipp_r);
	std::swap(Tip_l, Tipp_l);

	Tip_r = Ti_r;
	Tip_l = Ti_l;
      }

      return res;
    }
  }
}
