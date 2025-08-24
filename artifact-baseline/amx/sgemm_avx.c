/*
 * AVX kernel taken from dnnl_dump_cpu_jit_avx2_kernel_sgemm_kern.*.bin
 * Appears in oneDNN/examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp.
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

#define BLOCK_SIZE 4 // #inst sequences per block

float in[NUM_BLOCKS][3][ZMM_WIDTH_F32],
    in_z6_7[BLOCK_SIZE + 1][2][XMM_WIDTH_F32];
float offset[NUM_ZMM][ZMM_WIDTH_F32]; // only [8..31] used
float out[NUM_ZMM][ZMM_WIDTH_F32];    // zmm[8..31] are outputs

__m512 zmm[NUM_ZMM];

void __attribute__((always_inline)) inline avx_kernel() {
#pragma GCC unroll 24
  for (int i = 8; i < NUM_ZMM; i++)
    zmm[i] = _mm512_setzero(); // vpxorq
  zmm[6] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z6_7[BLOCK_SIZE][0]));
  zmm[7] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z6_7[BLOCK_SIZE][1]));

  PRAGMA_UNROLL(GCC unroll NUM_BLOCKS)
  for (int i = 0; i < NUM_BLOCKS; i++) {
    zmm[0] = _mm512_loadu_ps(in[i][0]);
    zmm[1] = _mm512_loadu_ps(in[i][1]);
    zmm[2] = _mm512_loadu_ps(in[i][2]);
    PRAGMA_UNROLL(GCC unroll BLOCK_SIZE)
    for (int j = 0; j < BLOCK_SIZE; j++) {
      zmm[8 + (2 * j)] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8 + (2 * j)]);
      zmm[16 + (2 * j)] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[16 + (2 * j)]);
      zmm[24 + (2 * j)] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[24 + (2 * j)]);
      zmm[6] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z6_7[j][0]));

      zmm[9 + (2 * j)] = _mm512_fmadd_ps(zmm[0], zmm[7], zmm[9 + (2 * j)]);
      zmm[17 + (2 * j)] = _mm512_fmadd_ps(zmm[1], zmm[7], zmm[17 + (2 * j)]);
      zmm[25 + (2 * j)] = _mm512_fmadd_ps(zmm[2], zmm[7], zmm[25 + (2 * j)]);
      zmm[7] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z6_7[j][1]));
    }
  }

#pragma GCC unroll 8
  for (int i = 0; i < 8; i++) {
    zmm[8 + i] = _mm512_add_ps(_mm512_loadu_ps(offset[8 + i]), zmm[8 + i]);
    zmm[16 + i] = _mm512_add_ps(_mm512_loadu_ps(offset[16 + i]), zmm[16 + i]);
    zmm[24 + i] = _mm512_add_ps(_mm512_loadu_ps(offset[24 + i]), zmm[24 + i]);

    _mm512_storeu_ps(out[8 + i], zmm[8 + i]);
    zmm[8 + i] = _mm512_setzero(); // vpxorq
    _mm512_storeu_ps(out[16 + i], zmm[16 + i]);
    zmm[16 + i] = _mm512_setzero();
    _mm512_storeu_ps(out[24 + i], zmm[24 + i]);
    zmm[24 + i] = _mm512_setzero();
  }
}

int main() {
  clock_t t[4];
  size_t hbm_size = 0;

  t[0] = clock();

  // initialize inputs with random values
  for (int i = 0; i < NUM_BLOCKS * 3 * ZMM_WIDTH_F32; i++) {
    ((float *)in)[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
  }
  for (int i = 0; i < (BLOCK_SIZE + 1) * 2 * XMM_WIDTH_F32; i++) {
    ((float *)in_z6_7)[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
  }
  for (int i = 0; i < NUM_ZMM * ZMM_WIDTH_F32; i++) {
    ((float *)offset)[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
  }

  int shape_in[] = {NUM_BLOCKS, 3, 16};
  int shape_in_z6_7[] = {5, 2, 4};
  int shape_offset[] = {32, 16};

  print_array_f32("in", (float *)in, 3, shape_in);
  print_array_f32("in_z6_7", (float *)in_z6_7, 3, shape_in_z6_7);
  print_array_f32("offset", (float *)offset, 2, shape_offset);

  t[1] = clock();

  // call kernel
  avx_kernel();

  t[2] = clock();

  // print outputs
  for (int i = 8; i < NUM_ZMM; i++) {
    for (int j = 0; j < ZMM_WIDTH_F32; j++)
      printf("%f ", out[i][j]);
    printf("\n");
  }

  int shape_out[] = {32, 16};
  print_array_f32("out", (float *)out, 2, shape_out);

  t[3] = clock();

  hbm_size += sizeof in + sizeof in_z6_7 + sizeof offset + sizeof out;
  printf("hbm size: %lu\n", hbm_size);
  print_elapsed_time(t);

  return 0;
}
