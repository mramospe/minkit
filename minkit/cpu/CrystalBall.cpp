#include <cmath>

// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001
// sqrt(2)
#define SQRT2 1.4142135623730951

extern "C" {
  /// Function shared by "normalization" and "evaluate_binned"
  double integral( double c, double s, double a, double n, double xmin, double xmax ) {

    double result = 0.;

    bool use_log = ( std::abs(n - 1.) < 1e-5 );

    double tmin = (xmin - c) / s;
    double tmax = (xmax - c) / s;

    if ( a < 0 ) {

      double tmp = tmin;
      tmin = -tmax;
      tmax = -tmp;
    }

    double aa = std::abs(a);

    if ( tmin >= -aa )
      result += s * SQRTPI_2 * (std::erf(tmax / SQRT2) - std::erf(tmin / SQRT2));
    else {
      double A = std::pow(n / aa, n) * std::exp(-0.5 * aa * aa);
      double B = n / aa - aa;

      if ( tmax <= -aa ) {

	if ( use_log )
	  result += A * s * (std::log(B - tmin) - std::log(B - tmax));
	else
	  result += A * s / (1. - n) * (1. / std::pow(B - tmin, n - 1.) - 1. / std::pow(B - tmax, n - 1.));
      }
      else {
	double t1 = 0.;

	if ( use_log )
	  t1 = A * s * (std::log(B - tmin) - std::log(n / aa));
	else
	  t1 = A * s / (1. - n) * (1. / std::pow(B - tmin, n - 1.) - 1. / std::pow(n / aa, n - 1.));

	double t2 = s * SQRTPI_2 * (std::erf(tmax / SQRT2) - std::erf(-aa / SQRT2));

	result += t1 + t2;
      }
    }

    return result != 0. ? result : 1e-300;
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, double c, double s, double a, double n ) {

    double t = ( a < 0 ? -1 : +1 ) * ( x - c ) / s;

    double aa = std::abs(a);

    if ( t >= -aa )
      return std::exp(-0.5 * t * t);
    else {
      double A = std::pow(n / aa, n) * std::exp(-0.5 * aa * aa);
      double B = n / aa - aa;

      return A / std::pow(B - t, n);
    }
  }

  /// Function for single-value
  double function( double x, double c, double s, double a, double n ) {
    return shared_function(x, c, s, a, n);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double* in, double c, double s, double a, double n ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], c, s, a, n);
  }

  /// Evaluate on a binned data set
  void evaluate_binned( int len, double *out, double c, double s, double a, double n, int nedges, int gap, double *edges )
  {

    for ( int i = 0; i < len; ++i ) {

      int ie = (i / gap) % nedges;

      out[i] = integral(c, s, a, n, edges[ie], edges[ie + 1]);
    }
  }

  /// Normalization
  double normalization( double c, double s, double a, double n, double xmin, double xmax ) {
    return integral(c, s, a, n, xmin, xmax);
  }
}
