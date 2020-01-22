#include <cmath>

extern "C" {
  /// Swap two variables
  static void swap( double &a, double &b ) {

    double tmp = a;
    a = b;
    b = tmp;
  }

  /// Function shared by "normalization" and "evaluate_binned"
  double integral( int n, double *p, double xmin, double xmax ) {

    if ( n == 0 )
      return (xmax - xmin);
    else if ( n == 1 )
      return (xmax - xmin) + 0.5 * p[0] * (xmax * xmax - xmin * xmin);
    else {

      double Tipp_r = 1.;
      double Tipp_l = 1.;

      double Tip_r = xmax;
      double Tip_l = xmin;

      double res = (xmax - xmin) + 0.5 * p[0] * (xmax * xmax - xmin * xmin);

      for ( int i = 1; i < n; ++i ) {

	double Ti_r = 2. * xmax * Tip_r - Tipp_r;
	double Tin_r = 2. * xmax * Ti_r - Tip_r;

	double Ti_l = 2. * xmin * Tip_l - Tipp_l;
	double Tin_l = 2. * xmin * Ti_l - Tip_l;

	res += 0.5 * p[i] * ((Tin_r - Tin_l) / (i + 1) - (Tip_r - Tip_l) / (i - 1));

	swap(Tip_r, Tipp_r);
	swap(Tip_l, Tipp_l);

	Tip_r = Ti_r;
	Tip_l = Ti_l;
      }

      return res;
    }
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, int n, double *p ) {

    if ( n == 0 )
      return 1.;
    else if ( n == 1 )
      return 1. + p[0] * x;
    else {

      double Tipp = 1.;
      double Tip = x;

      double res = p[0] * Tip + Tipp;

      for ( int i = 1; i < n; ++i ) {

	double Ti = 2. * x * Tip - Tipp;

	res += p[i] * Ti;

	swap(Tip, Tipp);

	Tip = Ti;
      }

      return res;
    }
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
