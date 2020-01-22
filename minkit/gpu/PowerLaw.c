/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double n )
{
  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  out[idx] = 1. / pow(fabs(x - c), n);
}

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, double c, double n, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  bool use_log = ( fabs(n - 1.) < 1e-5 );

  double result = 0.;

  if ( use_log )
    result = (log(xmax - c) - log(xmin - c));
  else
    result = 1. / (1. - n) * (1. / pow(fabs(xmax - c), n - 1.) - 1. / pow(fabs(xmin - c), n - 1.));

  out[idx] = result != 0. ? result : 1e-300;
}
