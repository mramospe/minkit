/** Functions to generate random samples.
 *
 * Placeholders are solved by reikna.
 */
#define IN_BIJECTION_MODULE(name) $bijection_module##name
#define IN_SAMPLER_MODULE(name) $sampler_module##name
#define IN_KEYGEN_MODULE(name) $keygen_module##name

/// Generate a random sample between 0 and 1
KERNEL void generate(int lgth, GLOBAL_MEM double *dest, int ctr_start) {
  const SIZE_T idx = get_global_id(0);

  if (idx >= lgth)
    return;

  IN_BIJECTION_MODULE(Key) key = IN_KEYGEN_MODULE(key_from_int)(idx);
  IN_BIJECTION_MODULE(Counter)
  ctr = IN_BIJECTION_MODULE(make_counter_from_int)(ctr_start);
  IN_BIJECTION_MODULE(State) st = IN_BIJECTION_MODULE(make_state)(key, ctr);

  dest[idx] = IN_SAMPLER_MODULE(sample)(&st).v[0];
}
