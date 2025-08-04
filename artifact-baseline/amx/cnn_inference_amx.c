/*
 * AMX kernel taken from dnnl_dump_cpu_jit_brgemm_amx_uker_base_t.*.bin
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

int8_t in_t4[TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)],
    in_t5[TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)], in_t6[0x800 * NUM_BLOCKS];
int8_t *in_t7 = in_t6 + 0x40;
int32_t out[4][TILE_SIZE_I32]; // tmm[0..3] are outputs

void __attribute__((always_inline)) amx_kernel(struct __tile_config *cfg) {
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
    _tile_loadd(5, in_t5 + i * 0x40, MAX_COLSB);
    _tile_dpbusd(1, 4, 7);
    _tile_dpbusd(2, 5, 6);
    _tile_dpbusd(3, 5, 7);
  }

  _tile_stored(0, out[0], MAX_COLSB);
  _tile_stored(1, out[1], MAX_COLSB);
  _tile_stored(2, out[2], MAX_COLSB);
  _tile_stored(3, out[3], MAX_COLSB);
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

  // initialize inputs with random values
  for (int i = 0; i < TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1); i++) {
    in_t4[i] = (int8_t)(rand() % 6 - 3);
    in_t5[i] = (int8_t)(rand() % 6 - 3);
  }
  for (int i = 0; i < 0x800 * NUM_BLOCKS; i++) {
    in_t6[i] = (int8_t)(rand() % 6 - 3);
  }
  // memset(in_t4, 2, TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1));
  // memset(in_t5, 3, TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1));
  // memset(in_t6, 4, 0x800 * NUM_BLOCKS);

  int shape_in_t4[] = {TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)};
  int shape_in_t5[] = {TILE_SIZE_I8 + 0x40 * (NUM_BLOCKS - 1)};
  int shape_in_t6[] = {0x800 * NUM_BLOCKS};

  print_array_i8("in_t4", in_t4, 1, shape_in_t4);
  print_array_i8("in_t5", in_t5, 1, shape_in_t5);
  print_array_i8("in_t6", in_t6, 1, shape_in_t6);

  t[1] = clock();

  // call kernel
  amx_kernel(&tile_cfg);

  t[2] = clock();

  // print outputs
  for (int i = 0; i < TILE_SIZE_I32; i++)
    printf("%d %d %d %d\n", out[0][i], out[1][i], out[2][i], out[3][i]);

  int shape_out[] = {4, 256};
  print_array_i32("out", (int32_t *)out, 2, shape_out);

  t[3] = clock();

  hbm_size += sizeof in_t4 + sizeof in_t5 + sizeof in_t6 + sizeof out;
  printf("hbm size: %lu\n", hbm_size);
  print_elapsed_time(t);

  return 0;
}
