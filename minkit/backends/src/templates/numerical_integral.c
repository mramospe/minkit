/****************************************
 * MIT License
 *
 * Copyright (c) 2020 Miguel Ramos Pernas
 ****************************************/

#ifdef USE_CPU
#include "Python.h"
#include "math.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_vegas.h>

#if NDIM == 1
/// Function proxy to integrate using 1-dimensional methods
double function_proxy(double x, void *vparams) {

  double *params = (double *)vparams;

  return FUNCTION(x, FWD_PARAMS(params)); // Definitions in "evaluators.c"
}
#endif

/// Function proxy to integrate using Monte Carlo methods
double monte_function_proxy(double *data, size_t, void *vparams) {

  double *params = (double *)vparams;

  return FUNCTION(DATA(data),
                  FWD_PARAMS(params)); // Definitions in "evaluators.c"
}

extern "C" {

#if NDIM == 1
/// Exposed function to integrate using the QNG method
PyObject *integrate_qng(double lb, double ub, PyObject *config,
                        double *params) {

  gsl_function func = {&function_proxy, params};

  double atol, rtol;
  PyArg_ParseTuple(config, "dd", &atol, &rtol);

  double res, err;
  size_t neval;
  gsl_integration_qng(&func, lb, ub, atol, rtol, &res, &err, &neval);

  return Py_BuildValue("(ddi)", res, err, neval);
}

/// Exposed function to integrate using the QAG method
PyObject *integrate_qag(double lb, double ub, PyObject *config,
                        double *params) {

  gsl_function func = {&function_proxy, params};

  double atol, rtol;
  int limit, key, workspace_size;
  PyArg_ParseTuple(config, "ddiii", &atol, &rtol, &limit, &key,
                   &workspace_size);

  gsl_integration_workspace *w =
      gsl_integration_workspace_alloc(workspace_size);

  double res, aerr;
  gsl_integration_qag(&func, lb, ub, atol, rtol, limit, key, w, &res, &aerr);

  gsl_integration_workspace_free(w);

  return Py_BuildValue("(dd)", res, aerr);
}

/// Exposed function to integrate using the CQUAD method
PyObject *integrate_cquad(double lb, double ub, PyObject *config,
                          double *params) {

  gsl_function func = {&function_proxy, params};

  double atol, rtol;
  int workspace_size;
  PyArg_ParseTuple(config, "ddi", &atol, &rtol, &workspace_size);

  gsl_integration_cquad_workspace *w =
      gsl_integration_cquad_workspace_alloc(workspace_size);

  double res, err;
  size_t neval;
  gsl_integration_cquad(&func, lb, ub, atol, rtol, w, &res, &err, &neval);

  gsl_integration_cquad_workspace_free(w);

  return Py_BuildValue("(ddi)", res, err, neval);
}
#endif

/// Exposed function to integrate using plain MonteCarlo
PyObject *integrate_plain(double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&monte_function_proxy, NDIM, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_plain_state *s = gsl_monte_plain_alloc(NDIM);

  int calls;
  PyArg_ParseTuple(config, "i", &calls);

  // Calculate the integral
  gsl_monte_plain_integrate(&func, lb, ub, NDIM, calls, r, s, &res, &err);

  gsl_monte_plain_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}

/// Exposed function to integrate using the MISER method
PyObject *integrate_miser(double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&monte_function_proxy, NDIM, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_miser_state *s = gsl_monte_miser_alloc(NDIM);

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
  gsl_monte_miser_integrate(&func, lb, ub, NDIM, calls, r, s, &res, &err);

  gsl_monte_miser_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}

/// Exposed function to integrate using the VEGAS method
PyObject *integrate_vegas(double *lb, double *ub, PyObject *config,
                          double *params) {

  gsl_monte_function func = {&monte_function_proxy, NDIM, params};

  double res, err;

  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  // Define the state
  gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(NDIM);

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
  gsl_monte_vegas_integrate(&func, lb, ub, NDIM, calls, r, s, &res, &err);

  do {
    gsl_monte_vegas_integrate(&func, lb, ub, NDIM, calls, r, s, &res, &err);

  } while (fabs(gsl_monte_vegas_chisq(s) - 1.0) > 0.5);

  gsl_monte_vegas_free(s);

  gsl_rng_free(r);

  return Py_BuildValue("(dd)", res, err);
}
}

#endif // USE_CPU
