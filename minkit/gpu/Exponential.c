/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double k )
{
  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  out[idx] = exp(k * x);
}

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, double k, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  out[idx] = 1. / k * (exp(k * xmax) - exp(k * xmin));
}
