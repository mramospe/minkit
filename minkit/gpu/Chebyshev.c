/** Definition of a Chebyshev polynomial PDF.
 */
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int n, GLOBAL_MEM double *p ) {

  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  if ( n == 0 ) {
    out[idx] = 1.;
    return;
  }
  else if ( n == 1 ) {
    out[idx] = 1. + p[0] * x;
    return;
  }
  else {

    double Tipp = 1.;
    double Tip = x;

    double res = p[0] * Tip + Tipp;

    for ( int i = 1; i < n; ++i ) {

      double Ti = 2. * x * Tip - Tipp;

      res += p[i] * Ti;

      Tipp = Tip;
      Tip  = Ti;
    }

    out[idx] = res;
  }
}
