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

#define CHECK_RESULT 0
#define OUTPUT 0

#define USE_RELU false

void print_matrix(char *name, elem_t *x, size_t dimI, size_t dimJ) {
  printf("DATA: %s %d %d\n", name, dimI, dimJ);
  for (size_t i = 0; i < dimI; ++i) {
    printf("DATA: ");
    for (size_t j = 0; j < dimJ; ++j)
      printf("%d ", x[i * dimJ + j]);
    printf("\n");
  }
}

void full_printMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

void full_matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++){
    for (size_t c = 0; c < DIM; c++) {
        C_full[r][c] = 0;
      for (size_t k = 0; k < DIM; k++){
        C_full[r][c] += A[r][k]*B[k][c];
      }
    }
  }
}

int full_is_equal(elem_t *x, elem_t *y, size_t dimI, size_t dimJ) {
  for (size_t i = 0; i < dimI; ++i)
    for (size_t j = 0; j < dimJ; ++j)
      if (x[i * dimJ + j] != y[i * dimJ + j]) {
        printf("I: %d, J: %d, x: %d, y: %d\n", i, j, x[i * dimJ + j], y[i * dimJ + j]);
        return 0;
      }
  return 1;
}
typedef struct __attribute__((__packed__)) AccBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} AccBlock;

// maintain a stack of blocks corresponding to
// a stack alloc and free strategy
#define HEAP_SIZE 100000
#define N_ACC_BLOCKS (HEAP_SIZE / sizeof(AccBlock))
AccBlock ACC_BLOCKS[N_ACC_BLOCKS];
uint32_t gemm_acc_free_block;
uint32_t gemm_acc_malloc(long unsigned int size) {
  // must have two free metadata blocks and
  // this allocation must have > 0 size
  if(size == 0) return -1;
  if(gemm_acc_free_block >= N_ACC_BLOCKS) return -1;

  size            = (size + DIM - 1) / DIM;
  uint32_t i      = gemm_acc_free_block;

  uint32_t loc = 0;
  if(i > 0) {
      loc = ACC_BLOCKS[i-1].loc + ACC_BLOCKS[i-1].size;
  }
  ACC_BLOCKS[i].size  = size;
  ACC_BLOCKS[i].loc   = 0;
  ACC_BLOCKS[i].is_used = 1;
  gemm_acc_free_block = i+1;

  return (ACC_BLOCKS[i].loc | ((uint32_t)0x80000000));
}

void gemm_acc_free(uint32_t addr) {
  if( gemm_acc_free_block == 0 )
      return;
  addr = addr & (uint32_t)(0x7FFFFFFF);
  // first case: free-ing the top of the block-stack
  if( ACC_BLOCKS[gemm_acc_free_block-1].loc == addr ) {
      ACC_BLOCKS[gemm_acc_free_block-1].is_used = 0;

      // Then go through and release as many blocks
      // as we can
      for(int i=gemm_acc_free_block-1; i >= 0; i--) {
          if(ACC_BLOCKS[i].is_used)
              break; // loop termination
          // otherwise...
          gemm_acc_free_block = i;
      }
  // second case: find the freed block and mark it
  } else {
      for(int i=gemm_acc_free_block-1; i >= 0; i--) {
          if(ACC_BLOCKS[i].loc == addr) {
              ACC_BLOCKS[i].is_used = 0;
              break;
          }
      }
  }
  return;
}

typedef struct __attribute__((__packed__)) NewBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} NewBlock;

NewBlock BLOCKS[HEAP_SIZE / sizeof(NewBlock)];
uint32_t gemm_last_ptr;

void gemm_init_mem() {
  for(uint32_t i=0; i<sizeof(BLOCKS); i++)
      ((uint8_t*)BLOCKS)[i] = 0;
  gemm_last_ptr = 0;
}

uint32_t gemm_malloc(long unsigned int size) {
  if(size == 0) return -1;
  size = (size + DIM - 1) / DIM;
  int i;
  for(i=0; i < HEAP_SIZE / sizeof(NewBlock) && BLOCKS[i].size > 0; i++) {
      if(BLOCKS[i].is_used) continue;
      if(BLOCKS[i].size < size) continue;
      break;
  }
  if(BLOCKS[i].size == 0) {
      BLOCKS[i].loc = gemm_last_ptr;
      BLOCKS[i].size = size;
      BLOCKS[i].is_used = 1;
      gemm_last_ptr += size;
      return BLOCKS[i].loc;
  }

  BLOCKS[i].is_used = 1;
  return BLOCKS[i].loc;
}

void gemm_free(uint32_t addr) {
  for(int i=0; BLOCKS[i].size > 0; i++) {
      if(BLOCKS[i].is_used && BLOCKS[i].loc == addr) {
          BLOCKS[i].is_used = 0;
          return;
      }
  }
  return;
}

typedef struct matmul_Context { 

  struct ConfigLoad {
      int_fast32_t src_stride;
  } ConfigLoad;

  struct ConfigLoad_id1 {
      int_fast32_t src_stride;
  } ConfigLoad_id1;

  struct ConfigMatmul {
      bool done;
  } ConfigMatmul;

  struct ConfigLoad_id2 {
      int_fast32_t src_stride;
  } ConfigLoad_id2;

  struct ConfigStore {
      float scale;
      int_fast32_t dst_stride;
      bool act;
  } ConfigStore;

} matmul_Context;

// matmul_27(
//     scale : f32 @DRAM,
//     act : bool,
//     A : i8[784, 256] @DRAM,
//     B : i8[256, 1024] @DRAM,
//     C : i8[784, 1024] @DRAM
// )
void matmul_27( matmul_Context *ctxt, const float* scale, bool act, const int8_t* A, const int8_t* B, int8_t* C ) {
  gemmini_extended_config_st((1024), (act), (scale)[0]);

  gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);

  gemmini_extended3_config_ld((1024), 1.0f, 0, 2);

  gemmini_extended3_config_ld((256), 1.0f, 0, 1);

  gemmini_extended3_config_ld(0, 1.0f, 0, 0);

  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 16 * 4 * 8 * sizeof(int32_t)));
  int8_t *a = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 7 * sizeof(int8_t)));
  int8_t *b = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 4 * 8 * sizeof(int8_t)));
  for (int_fast32_t io = 0; io < 7; io++) {
    for (int_fast32_t jo = 0; jo < 2; jo++) {
      for (int_fast32_t i = 0; i < 7; i++) {
        for (int_fast32_t j = 0; j < 8; j++) {
          gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))),(16), (16) );
          gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))),(16), (16) );
          gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))),(16), (16) );
          gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))),(16), (16) );
          for (int_fast32_t ko = 0; ko < 4; ko++) {
            if (jo == 0) {
              if (j == 0) {
                gemmini_extended_mvin2( &A[(16 * i + 112 * io) * (256) + 64 * ko], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024))/16))), 16*(4), (16) );
              }
            }
            if (i == 0) {
              gemmini_extended_mvin3( &B[(64 * ko) * (1024) + 64 * j + 512 * jo], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096))/16))), 16*(4), (16) );
            }
            if (i == 0) {
              gemmini_extended_mvin3( &B[(16 + 64 * ko) * (1024) + 64 * j + 512 * jo], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 1024)/16))), 16*(4), (16) );
            }
            if (i == 0) {
              gemmini_extended_mvin3( &B[(32 + 64 * ko) * (1024) + 64 * j + 512 * jo], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (1024))/16))), 16*(4), (16) );
            }
            if (i == 0) {
              gemmini_extended_mvin3( &B[(48 + 64 * ko) * (1024) + 64 * j + 512 * jo], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (1024))/16))), 16*(4), (16) );
            }
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 1024)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 1024 + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 1024 + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + 1024 + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (2) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
            gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (16384) + (ko) * (4096) + (3) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (4096) + (ko) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
          }
          gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 112 * io) * (1024) + 64 * j + 512 * jo]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16)), (16), (16) );
          gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 112 * io) * (1024) + 16 + 64 * j + 512 * jo]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16)), (16), (16) );
          gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 112 * io) * (1024) + 32 + 64 * j + 512 * jo]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16)), (16), (16) );
          gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 112 * io) * (1024) + 48 + 64 * j + 512 * jo]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16)), (16), (16) );
        }
      }
    }
  }
  gemm_free((uint64_t)(b));
  gemm_free((uint64_t)(a));
  gemm_acc_free((uint32_t)(res));
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif


  static elem_t A[784][256] row_align(1);
  static elem_t B[256][1024] row_align(1);
  static elem_t C[784][1024] row_align(1);
  matmul_Context *ctxt;
  float c_scale = (float) ACC_SCALE_IDENTITY;

  for(int i =0 ; i < 784; i++){
    for(int j =0; j < 256; j++){
      A[i][j] = rand() % 2;
    }
  }
  for(int i =0 ; i < 256; i++){
    for(int j =0; j < 1024; j++){
      B[i][j] = rand() % 2;
    }
  }
#if CHECK_RESULT == 1
  static elem_t gold[784][1024] row_align(1);
  tiled_matmul_auto(784, 1024, 256,
    (elem_t*) A, (elem_t*) B, NULL, (elem_t*) gold,
    256, 1024, 1024, 1024,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
    false, false,
    false, true,
    0,
    WS);
#endif

  gemmini_fence();
  //unsigned long start = read_cycles();
  matmul_27(ctxt, &c_scale, 0, A, B, C );
  gemmini_fence();
  //unsigned long end = read_cycles();
  #if OUTPUT
  //printf("Starting matmul 27 exo lib with for loops\n");
  //printf("Cycles taken: %u\n", end-start);
  #else
  print_matrix("A", A, 784, 256);
  print_matrix("B", B, 256, 1024);
  print_matrix("C", C, 784, 1024);
  #endif

#if CHECK_RESULT == 1
    if(full_is_equal(gold, C, 784, 1024)){
      #if OUTPUT
      printf("Success\n");
      #endif
    }
    else{
      #if OUTPUT
      printf("Failed\n");
      #endif
    }
#endif

  exit(0);
}
