// See LICENSE for license details.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define I 1

#if (3 * I + 1) * DIM > BANK_ROWS * BANK_NUM
#error not enough scratchpad space
#endif

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

  static elem_t A[I * DIM][DIM] row_align(1);
  static elem_t B[DIM][DIM] row_align(1);
  static elem_t D[I * DIM][DIM] row_align(1);
  static elem_t C[I * DIM][DIM] row_align(1);

  size_t A_sp_addr = 0;
  size_t B_sp_addr = I * DIM;
  size_t D_sp_addr = (I + 1) * DIM;
  size_t C_sp_addr = (2 * I + 1) * DIM;

  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  for (size_t i = 0; i < I; i++) {
    gemmini_mvin(&A[i * DIM][0], A_sp_addr + i * DIM);
  }
  gemmini_mvin(&B[0][0], B_sp_addr);
  for (size_t i = 0; i < I; i++) {
    gemmini_mvin(&D[i * DIM][0], D_sp_addr + i * DIM);
  }

  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
  gemmini_preload(B_sp_addr, C_sp_addr);
  gemmini_compute_preloaded(A_sp_addr, D_sp_addr);
  for (size_t i = 1; i < I; i++) {
    gemmini_preload(GARBAGE_ADDR, C_sp_addr + i * DIM);
    gemmini_compute_accumulated(A_sp_addr + i * DIM, D_sp_addr + i * DIM);
  }

  for (size_t i = 0; i < I; i++) {
    gemmini_mvout(&C[i * DIM][0], C_sp_addr + i * DIM);
  }
  gemmini_fence();

  exit(0);
}
