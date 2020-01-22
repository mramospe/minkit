/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int n, GLOBAL_MEM double *p ) {

  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  if ( n == 0 ) {
    out[idx] = 1.;
    return;
  }

  double o = x * p[n - 1];
  for ( int i = 1; i < n; ++i )
    o = x * (o + p[n - i - 1]);

  out[idx] = o + 1.;
}

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, int n, GLOBAL_MEM double *p, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  if ( n == 0 )
    out[idx] = xmax - xmin;

  // Right integral
  double r = xmax * p[n - 1] / (n + 1);
  for ( int i = 1; i < n; ++i )
    r = xmax * (r + p[n - i - 1] / (n + 1 - i));

  // Left integral
  double l = xmin * p[n - 1] / (n + 1);
  for ( int i = 1; i < n; ++i )
    l = xmin * (l + p[n - i - 1] / (n + 1 - i));

  out[idx] = (1. + r) * xmax - (1. + l) * xmin;
}
