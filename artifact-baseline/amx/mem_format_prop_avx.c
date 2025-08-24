/*
 * AVX kernel taken from dnnl_dump_cpu_jit_brgemm_kernel_t.*.bin
 * Appears in oneDNN/examples/memory-format-propagation-cpp.
 */
#include "setup.h"

void print_array_f32(char *name, float *x, int num_dims, int *shape) {
  size_t total_len = 1;
  for (int i = 0; i < num_dims; i++) {
    total_len *= shape[i];
  }

  // Print simplified format: name dims shape dtype
  printf("DATA: %s %d ", name, num_dims);
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", shape[i]);
  }
  printf("f32\n");

  printf("DATA: ");
  for (size_t i = 0; i < total_len; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

#define BLOCK_SIZE 14 // #inst sequences per block

float in_z0[BLOCK_SIZE][XMM_WIDTH_F32], in_z2_3[NUM_BLOCKS][2][ZMM_WIDTH_F32];
float out[NUM_ZMM][ZMM_WIDTH_F32]; // zmm[4..31] are outputs

__m512 zmm[NUM_ZMM];

void __attribute__((always_inline)) inline avx_kernel() {
#pragma GCC unroll 28
  for (int i = NUM_ZMM - 1; i >= 4; i--)
    zmm[i] = _mm512_setzero(); // vpxorq

  PRAGMA_UNROLL(GCC unroll NUM_BLOCKS)
  for (int i = 0; i < NUM_BLOCKS; i++) {
    zmm[3] = _mm512_loadu_ps(in_z2_3[i][1]);
    zmm[2] = _mm512_loadu_ps(in_z2_3[i][0]);
    PRAGMA_UNROLL(GCC unroll BLOCK_SIZE)
    for (int j = 0; j < BLOCK_SIZE; j++) {
      zmm[0] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z0[j]));
      zmm[31 - (2 * j)] = _mm512_fmadd_ps(zmm[0], zmm[3], zmm[31 - (2 * j)]);
      zmm[30 - (2 * j)] = _mm512_fmadd_ps(zmm[0], zmm[2], zmm[30 - (2 * j)]);
    }
  }

#pragma GCC unroll 28
  for (int i = NUM_ZMM - 1; i >= 4; i--)
    _mm512_storeu_ps(out[i], zmm[i]);
}

int main() {
  clock_t t[4];
  size_t hbm_size = 0;

  t[0] = clock();

  // initialize inputs with random values
  for (int i = 0; i < BLOCK_SIZE * XMM_WIDTH_F32; i++) {
    ((float *)in_z0)[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
  }
  for (int i = 0; i < NUM_BLOCKS * 2 * ZMM_WIDTH_F32; i++) {
    ((float *)in_z2_3)[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
  }

  int shape_in_z0[] = {14, 4};
  int shape_in_z2_3[] = {NUM_BLOCKS, 2, 16};

  print_array_f32("in_z0", (float *)in_z0, 2, shape_in_z0);
  print_array_f32("in_z2_3", (float *)in_z2_3, 3, shape_in_z2_3);

  t[1] = clock();

  // call kernel
  avx_kernel();

  t[2] = clock();

  // print outputs
  for (int i = 4; i < NUM_ZMM; i++) {
    for (int j = 0; j < ZMM_WIDTH_F32; j++)
      printf("%f ", out[i][j]);
    printf("\n");
  }

  int shape_out[] = {32, 16};
  print_array_f32("out", (float *)out, 2, shape_out);

  t[3] = clock();

  hbm_size += sizeof in_z0 + sizeof in_z2_3 + sizeof out;
  printf("hbm size: %lu\n", hbm_size);
  print_elapsed_time(t);

  return 0;
}
