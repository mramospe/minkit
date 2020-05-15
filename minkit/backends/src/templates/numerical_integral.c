#ifdef USE_CPU
#include "Python.h"
#include "math.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_vegas.h>
#include <stdio.h>
/// Function proxy to integrate
double function_proxy(double *data, size_t dim, void *vparams) {

  (void)(dim); // suppress warnings for unused parameters

  double *params = (double *)vparams;

  return FUNCTION(DATA(data),
                  FWD_PARAMS(params)); // Definitions in "evaluators.c"
}

extern "C" {

/// Exposed function to integrate using plain MonteCarlo
PyObject *integrate_plain(size_t dim, double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&function_proxy, dim, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_plain_state *s = gsl_monte_plain_alloc(dim);

  size_t calls;
  PyArg_ParseTuple(config, "i", &calls);

  // Calculate the integral
  gsl_monte_plain_integrate(&func, lb, ub, dim, calls, r, s, &res, &err);

  gsl_monte_plain_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}

/// Exposed function to integrate using the MISER method
PyObject *integrate_miser(size_t dim, double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&function_proxy, dim, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_miser_state *s = gsl_monte_miser_alloc(dim);

  int calls;
  double estimate_frac;
  int min_calls;
  int min_calls_per_bisection;
  double alpha;
  double dither;

  PyArg_ParseTuple(config, "idiidd", &calls, &estimate_frac, &min_calls,
                   &min_calls_per_bisection, &alpha, &dither);

  gsl_monte_miser_params p;
  gsl_monte_miser_params_get(s, &p);
  p.estimate_frac = estimate_frac;
  p.min_calls = min_calls;
  p.min_calls_per_bisection = min_calls_per_bisection;
  p.alpha = alpha;
  p.dither = dither;
  gsl_monte_miser_params_set(s, &p);

  // Calculate the integral
  gsl_monte_miser_integrate(&func, lb, ub, dim, calls, r, s, &res, &err);

  gsl_monte_miser_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}

/// Exposed function to integrate using the VEGAS method
PyObject *integrate_vegas(size_t dim, double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&function_proxy, dim, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(dim);

  int calls;
  double alpha;
  int iterations;
  int mode;

  PyArg_ParseTuple(config, "idii", &calls, &alpha, &iterations, &mode);

  gsl_monte_vegas_params p;
  gsl_monte_vegas_params_get(s, &p);
  p.alpha = alpha;
  p.iterations = iterations;
  p.mode = mode;
  gsl_monte_vegas_params_set(s, &p);

  // Calculate the integral
  gsl_monte_vegas_integrate(&func, lb, ub, dim, calls, r, s, &res, &err);

  do {
    gsl_monte_vegas_integrate(&func, lb, ub, dim, calls, r, s, &res, &err);

  } while (fabs(gsl_monte_vegas_chisq(s) - 1.0) > 0.5);

  gsl_monte_vegas_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}
}

#endif // USE_CPU
