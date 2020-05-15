/******************************************************************************
 * Definition of the template to build a function in both CPU and GPU backends.
 ******************************************************************************/
#ifdef USE_CPU
extern "C" {
#endif

WITHIN_KERNEL double pdf_function($function_arguments) { $function_code; }

#define FUNCTION pdf_function

#ifdef USE_CPU
}
#endif
