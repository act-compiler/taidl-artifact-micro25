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

void print_matrix(char *name, elem_t *x, size_t dimI, size_t dimJ) {
  printf("DATA: %s %d %d\n", name, dimI, dimJ);
  for (size_t i = 0; i < dimI; ++i) {
    printf("DATA: ");
    for (size_t j = 0; j < dimJ; ++j)
      printf("%d ", x[i * dimJ + j]);
    printf("\n");
  }
}

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

  // Generate 5 data sets with different random input ranges
  int ranges[5][2] = {
    {-3, 3},
    {-7, 7},
    {-15, 15},
    {-31, 31},
    {-63, 63}
  };

  for (int data_set = 0; data_set < 5; data_set++) {
    int range_min = ranges[data_set][0];
    int range_max = ranges[data_set][1];
    int range_size = range_max - range_min + 1;

    for (size_t i = 0; i < I * DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        A[i][j] = (int8_t)(rand() % range_size) + range_min;

    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        B[i][j] = (int8_t)(rand() % range_size) + range_min;

    for (size_t i = 0; i < I * DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        D[i][j] = (int8_t)(rand() % range_size) + range_min;

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

    gemmini_config_ex(OUTPUT_STATIONARY, 0, 0);
    for (size_t i = 0; i < I; i++) {
      gemmini_preload(D_sp_addr + i * DIM, C_sp_addr + i * DIM);
      gemmini_compute_preloaded(A_sp_addr + i * DIM, B_sp_addr);
    }

    for (size_t i = 0; i < I; i++) {
      gemmini_mvout(&C[i * DIM][0], C_sp_addr + i * DIM);
    }
    gemmini_fence();

    print_matrix("A", &A[0][0], I * DIM, DIM);
    print_matrix("B", &B[0][0], DIM, DIM);
    print_matrix("D", &D[0][0], I * DIM, DIM);
    print_matrix("C", &C[0][0], I * DIM, DIM);
  }

  exit(0);
}
