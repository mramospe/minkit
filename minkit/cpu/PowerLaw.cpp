#include <cmath>

extern "C" {
  /// Function shared by "normalization" and "evaluate_binned"
  double integral( double c, double n, double xmin, double xmax ) {

    bool use_log = ( std::abs(n - 1.) < 1e-5 );

    double result = 0.;

    if ( use_log )
      result = (std::log(xmax - c) - std::log(xmin - c));
    else
      result = 1. / (1. - n) * (1. / std::pow(std::abs(xmax - c), n - 1.) - 1. / std::pow(std::abs(xmin - c), n - 1.));

    return result != 0. ? result : 1e-300;
  }

  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, double c, double n ) {

    return 1. / std::pow(std::abs(x - c), n);
  }

  /// Function for single-value
  double function( double x, double c, double n ) {
    return shared_function(x, c, n);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double* in, double c, double n ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], c, n);
  }

  /// Evaluate on a binned data set
  void evaluate_binned( int len, double *out, double c, double n, int nedges, int gap, double *edges )
  {

    for ( int i = 0; i < len; ++i ) {

      int ie = (i / gap) % nedges;

      out[i] = integral(c, n, edges[ie], edges[ie + 1]);
    }
  }

  /// Normalization
  double normalization( double c, double n, double xmin, double xmax ) {
    return integral(c, n, xmin, xmax);
  }
}
