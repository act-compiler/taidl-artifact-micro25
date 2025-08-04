/*
 * AMX-AVX kernel taken from dnnl_dump_cpu_jit_brgemm_amx_uker_base_t.*.bin
 * Appears in oneDNN/examples/cnn-inference-int8-cpp.
 */
#include "setup.h"

void print_array_i8(char *name, int8_t *x, int num_dims, int *shape) {
  size_t total_len = 1;
  for (int i = 0; i < num_dims; i++) {
    total_len *= shape[i];
  }

  // Print simplified format: name dims shape dtype
  printf("DATA: %s %d ", name, num_dims);
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", shape[i]);
  }
  printf("i8\n");

  printf("DATA: ");
  for (size_t i = 0; i < total_len; i++) {
    printf("%d ", x[i]);
  }
  printf("\n");
}

void print_array_i32(char *name, int32_t *x, int num_dims, int *shape) {
  size_t total_len = 1;
  for (int i = 0; i < num_dims; i++) {
    total_len *= shape[i];
  }

  // Print simplified format: name dims shape dtype
  printf("DATA: %s %d ", name, num_dims);
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", shape[i]);
  }
  printf("i32\n");

  printf("DATA: ");
  for (size_t i = 0; i < total_len; i++) {
    printf("%d ", x[i]);
  }
  printf("\n");
}

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

void print_array_u8(char *name, uint8_t *x, int num_dims, int *shape) {
  size_t total_len = 1;
  for (int i = 0; i < num_dims; i++) {
    total_len *= shape[i];
  }

  // Print simplified format: name dims shape dtype
  printf("DATA: %s %d ", name, num_dims);
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", shape[i]);
  }
  printf("u8\n");

  printf("DATA: ");
  for (size_t i = 0; i < total_len; i++) {
    printf("%d ", x[i]);
  }
  printf("\n");
}

// AMX
int8_t in_t4[TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)],
    in_t5[TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)], in_t6[0x800 * NUM_BLOCKS];
int8_t *in_t7 = in_t6 + 0x40;
int32_t amx_out[4][TILE_SIZE_I32]; // tmm[0..3] are outputs

// AVX
float max[ZMM_WIDTH_F32], in_z0[XMM_WIDTH_F32];
float in_z10[ZMM_WIDTH_F32], in_z15[ZMM_WIDTH_F32],
    avx_in[NUM_BLOCKS][3][ZMM_WIDTH_F32];
uint8_t avx_out[NUM_BLOCKS][3][ZMM_WIDTH_F32]; // zmm[29..31] are outputs

__m512 zmm[NUM_ZMM];

void __attribute__((always_inline)) amx_avx_kernel(struct __tile_config *cfg) {
  __m512i zmm_i32;
  __mmask16 k2 = _cvtu32_mask16(0xffff);
  zmm[10] = _mm512_loadu_ps(in_z10);
  zmm[15] = _mm512_loadu_ps(in_z15);
  zmm[8] = _mm512_setzero(); // vpxorq
  __m128 xmm = (__m128)_mm_cvtsi64_si128(0x437f0000);
  zmm[9] = _mm512_broadcastss_ps(xmm); // broadcast 255.0 to all elements

  _tile_loadconfig(cfg);
  _tile_zero(0);
  _tile_zero(1);
  _tile_zero(2);
  _tile_zero(3);

  PRAGMA_UNROLL(GCC unroll NUM_BLOCKS)
  for (int i = 0; i < NUM_BLOCKS; i++) {
    _tile_loadd(4, in_t4 + i * 0x40, MAX_COLSB);
    _tile_loadd(6, in_t6 + i * 0x800, MAX_COLSB);
    _tile_loadd(7, in_t7 + i * 0x800, MAX_COLSB);
    _tile_dpbusd(0, 4, 6);

    zmm[31] = _mm512_loadu_ps(avx_in[i][2]); // load 16 x f32
    zmm[30] = _mm512_loadu_ps(avx_in[i][1]);
    zmm[29] = _mm512_loadu_ps(avx_in[i][0]);

    zmm[31] = _mm512_mul_ps(zmm[15], zmm[31]);
    zmm[30] = _mm512_mul_ps(zmm[15], zmm[30]);
    zmm[29] = _mm512_mul_ps(zmm[15], zmm[29]);

    zmm[31] = _mm512_add_ps(zmm[10], zmm[31]);
    zmm[30] = _mm512_add_ps(zmm[10], zmm[30]);
    zmm[29] = _mm512_add_ps(zmm[10], zmm[29]);

    zmm[29] = _mm512_max_ps(_mm512_loadu_ps(max), zmm[29]);
    zmm[30] = _mm512_max_ps(_mm512_loadu_ps(max), zmm[30]);
    zmm[31] = _mm512_max_ps(_mm512_loadu_ps(max), zmm[31]);

    zmm[0] = _mm512_broadcastss_ps(_mm_loadu_ps(in_z0));
    zmm[31] = _mm512_mul_ps(zmm[0], zmm[31]);
    zmm[30] = _mm512_mul_ps(zmm[0], zmm[30]);
    zmm[29] = _mm512_mul_ps(zmm[0], zmm[29]);

    zmm[31] = _mm512_max_ps(zmm[8], zmm[31]);
    zmm[31] = _mm512_min_ps(zmm[9], zmm[31]);
    zmm_i32 = _mm512_cvtps_epi32(zmm[31]); // convert 16 x f32 --> 16 x i32
    _mm512_mask_cvtusepi32_storeu_epi8(avx_out[i][2], k2,
                                       zmm_i32); // convert to 16 x u8 and store
    _tile_loadd(5, in_t5 + i * 0x40, MAX_COLSB);
    _tile_dpbusd(1, 4, 7);

    zmm[30] = _mm512_max_ps(zmm[8], zmm[30]);
    zmm[30] = _mm512_min_ps(zmm[9], zmm[30]);
    zmm_i32 = _mm512_cvtps_epi32(zmm[30]);
    _mm512_mask_cvtusepi32_storeu_epi8(avx_out[i][1], k2, zmm_i32);
    _tile_dpbusd(2, 5, 6);

    zmm[29] = _mm512_max_ps(zmm[8], zmm[29]);
    zmm[29] = _mm512_min_ps(zmm[9], zmm[29]);
    zmm_i32 = _mm512_cvtps_epi32(zmm[29]);
    _mm512_mask_cvtusepi32_storeu_epi8(avx_out[i][0], k2, zmm_i32);
    _tile_dpbusd(3, 5, 7);
  }

  _tile_stored(0, amx_out[0], MAX_COLSB);
  _tile_stored(1, amx_out[1], MAX_COLSB);
  _tile_stored(2, amx_out[2], MAX_COLSB);
  _tile_stored(3, amx_out[3], MAX_COLSB);
  _tile_release();
}

int main() {
  clock_t t[4];
  size_t hbm_size = 0;
  struct __tile_config tile_cfg = {0};

  if (!set_tiledata_use())
    return -1;
  init_tile_config(&tile_cfg);

  t[0] = clock();

  // initialize AMX inputs with random values
  for (int i = 0; i < TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1); i++) {
    in_t4[i] = (int8_t)(rand() % 6 - 3);
    in_t5[i] = (int8_t)(rand() % 6 - 3);
  }
  for (int i = 0; i < 0x800 * NUM_BLOCKS; i++) {
    in_t6[i] = (int8_t)(rand() % 6 - 3);
  }
  // initialize AVX inputs with random values
  for (int i = 0; i < ZMM_WIDTH_F32; i++) {
    max[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
  for (int i = 0; i < XMM_WIDTH_F32; i++) {
    in_z0[i] = ((float)rand() / RAND_MAX) * 3.0f - 1.0f;
  }
  for (int i = 0; i < ZMM_WIDTH_F32; i++) {
    in_z10[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    in_z15[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
  for (int i = 0; i < NUM_BLOCKS * 3 * ZMM_WIDTH_F32; i++) {
    ((float *)avx_in)[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  int shape_in_t4[] = {TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)};
  int shape_in_t5[] = {TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)};
  int shape_in_t6[] = {0x800 * NUM_BLOCKS};
  int shape_max[] = {ZMM_WIDTH_F32};
  int shape_in_z0[] = {XMM_WIDTH_F32};
  int shape_in_z10[] = {ZMM_WIDTH_F32};
  int shape_in_z15[] = {ZMM_WIDTH_F32};
  int shape_avx_in[] = {NUM_BLOCKS, 3, 16};

  print_array_i8("in_t4", in_t4, 1, shape_in_t4);
  print_array_i8("in_t5", in_t5, 1, shape_in_t5);
  print_array_i8("in_t6", in_t6, 1, shape_in_t6);
  print_array_f32("max", max, 1, shape_max);
  print_array_f32("in_z0", in_z0, 1, shape_in_z0);
  print_array_f32("in_z10", in_z10, 1, shape_in_z10);
  print_array_f32("in_z15", in_z15, 1, shape_in_z15);
  print_array_f32("avx_in", (float *)avx_in, 3, shape_avx_in);

  t[1] = clock();

  // call kernel
  amx_avx_kernel(&tile_cfg);

  t[2] = clock();

  // print AMX outputs
  for (int i = 0; i < TILE_SIZE_I32; i++)
    printf("%d %d %d %d\n", amx_out[0][i], amx_out[1][i], amx_out[2][i],
           amx_out[3][i]);
  // print AVX outputs
  for (int i = 0; i < NUM_BLOCKS; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < ZMM_WIDTH_F32; k++)
        printf("%d ", avx_out[i][j][k]);
    }
    printf("\n");
  }

  int shape_out1[] = {4, 256};
  int shape_out2[] = {NUM_BLOCKS, 3, 16};

  print_array_i32("out1", (int32_t *)amx_out, 2, shape_out1);
  print_array_u8("out2", (uint8_t *)avx_out, 3, shape_out2);

  t[3] = clock();

  hbm_size += sizeof in_t4 + sizeof in_t5 + sizeof in_t6 + sizeof amx_out;
  hbm_size += sizeof max + sizeof in_z0 + sizeof in_z10 + sizeof in_z15 +
              sizeof avx_in + sizeof avx_out;
  printf("hbm size: %lu\n", hbm_size);
  print_elapsed_time(t);

  return 0;
}
