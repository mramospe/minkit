/// Evaluate on an unbinned data set
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

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, int n, GLOBAL_MEM double *p, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  if ( n == 0 )
    out[idx] = (xmax - xmin);
  else if ( n == 1 )
    out[idx] = (xmax - xmin) + 0.5 * p[0] * (xmax * xmax - xmin * xmin);
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

      double tmp;

      // Swap
      tmp = Tip_r;
      Tip_r = Tipp_r;
      Tipp_r = tmp;

      // Swap
      tmp    = Tip_l;
      Tip_l  = Tipp_l;
      Tipp_l = tmp;

      Tip_r = Ti_r;
      Tip_l = Ti_l;
    }

    out[idx] = res;
  }
}
