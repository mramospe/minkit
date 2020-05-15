/******************************************************************************
 * Definition of the template to build an integral function in both CPU and
 * GPU backends.
 ******************************************************************************/
#ifdef USE_CPU
extern "C" {
#endif

WITHIN_KERNEL double integral_function($integral_arguments) { $integral_code; }

#define INTEGRAL integral_function

#ifdef USE_CPU
}
#endif
