#include <cmath>

extern "C" {
  /// Function shared by "function" and "evaluate"
  static inline double shared_function( double x, double a, double theta, double alpha, double beta ) {
    double d = (x - a) / theta;
    return std::pow(d, alpha * beta - 1) * std::exp(- std::pow(d, beta));
  }

  /// Function for single-value
  double function( double x, double a, double theta, double alpha, double beta ) {
    return shared_function(x, a, theta, alpha, beta);
  }

  /// Evaluate on an unbinned data set
  void evaluate( int len, double *out, double* in, double a, double theta, double alpha, double beta ) {

    for ( int i = 0; i < len; ++i )
      out[i] = shared_function(in[i], a, theta, alpha, beta);
  }
}
