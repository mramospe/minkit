/****************************************
 * MIT License
 *
 * Copyright (c) 2020 Miguel Ramos Pernas
 ****************************************/

/******************************************************************************
 * Definition of the template to build a primitive function in both CPU and
 * GPU backends.
 ******************************************************************************/
#ifdef USE_CPU
extern "C" {
#endif

WITHIN_KERNEL double primitive_function($primitive_arguments) {
  $primitive_code;
}

WITHIN_KERNEL double integral_function($integral_arguments) {

  return primitive_function($primitive_fwd_args_max) -
         primitive_function($primitive_fwd_args_min);
}

#ifdef USE_CPU
}
#endif
