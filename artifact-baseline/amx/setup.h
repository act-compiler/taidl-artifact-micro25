#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

#define NUM_TMM 8
#define MAX_ROWS 16
#define MAX_COLSB 64
#define TILE_SIZE_I8 1024
#define TILE_SIZE_I32 256

#define NUM_ZMM 32
#define ZMM_WIDTH_I8 64
#define ZMM_WIDTH_F32 16
#define XMM_WIDTH_I8 16
#define XMM_WIDTH_F32 4

#define fill_float_array(arr, len, val)                                        \
  for (int i = 0; i < (len); i++)                                              \
  ((float *)arr)[i] = val

#if UNROLL == 1
#define DO_PRAGMA(x) _Pragma(#x)
#define PRAGMA_UNROLL(x) DO_PRAGMA(x)
#else
#define PRAGMA_UNROLL(x)
#endif

// Tile configuration data format.
struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[8];
  uint16_t reserved_1[8];
  uint8_t rows[8];
  uint8_t reserved_2[8];
};

// Set tile configuration.
void init_tile_config(struct __tile_config *cfg) {
  cfg->palette_id = 1;
  cfg->start_row = 0;

  for (int i = 0; i < 8; i++) {
    cfg->rows[i] = MAX_ROWS;
    cfg->colsb[i] = MAX_COLSB;
  }
}

// Request permission to use AMX.
bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\nFail to do XFEATURE_XTILEDATA\n\n");
    return false;
  }
  return true;
}

double get_elapsed_time(clock_t start, clock_t stop) {
  return (stop - start) * 1000.0 / CLOCKS_PER_SEC;
}

void print_elapsed_time(clock_t *t) {
  // printf("\n========== times (ms) ==========\n");
  // printf("prologue: %.3f\n", get_elapsed_time(t[0], t[1]));
  printf("kernel: %.3f\n", get_elapsed_time(t[1], t[2]));
  // printf("epilogue: %.3f\n", get_elapsed_time(t[2], t[3]));
}
