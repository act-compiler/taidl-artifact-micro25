#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#define LAYERS 12

#define NORM_STAT_IDS 4

static void sp_tiled_norm(const size_t I, const size_t J,
        const acc_t * in, elem_t * out,
        size_t A_row_stride, size_t C_row_stride,
        int act) {
#ifdef HAS_NORMALIZATIONS
    size_t A_blocks = (J/DIM + (J % DIM != 0));
    if (A_blocks > MAX_BLOCK_LEN_ACC) A_blocks = MAX_BLOCK_LEN_ACC;
    size_t C_blocks = (J/DIM + (J % DIM != 0));
    if (C_blocks > MAX_BLOCK_LEN) C_blocks = MAX_BLOCK_LEN;

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    const size_t rounded_up_J = (J / DIM + (J % DIM != 0)) * DIM;

    for (size_t i = 0; i < I; i += DIM) {
        // Mvin
        for (size_t j = 0; j < J; j += A_blocks * DIM) {
            const size_t cols = j + A_blocks*DIM <= J ? A_blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const acc_t * const A_dram_addr = in + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;

            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }

        // Mvout
        if (act == LAYERNORM) {
            uint32_t norm_cmds[][2] = {{1,2},{3,4},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = I - i < DIM ? I - i : DIM;
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                    NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                    for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                        gemmini_config_norm(0, 0, 0, 0, stat_id, 0, 0);
                        const size_t r = row + stat_id;
                        for (size_t jj = 0; jj < J; jj += C_blocks * DIM) {
                            uint32_t norm_C_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + jj + r;
                            if (jj + C_blocks*DIM >= J) {
                                norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                            } else {
                                norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                            }
                            void * const C_dram_addr = (int8_t*)out +
                                (i*C_row_stride + jj) * sizeof(elem_t) +
                                r * C_row_stride * sizeof(elem_t);
                            const size_t cols = J - jj < C_blocks * DIM ? J - jj : C_blocks * DIM;
                            gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                        }
                    }
                }
            }
        } else if (act == SOFTMAX) {
            uint32_t norm_cmds[][2] = {{5,5},{6,7},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = I - i < DIM ? I - i : DIM;
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                    NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                    for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                        // set stat id only
                        gemmini_config_norm(0, 0, 1, 0, stat_id, 0, 0);
                        const size_t r = row + stat_id;
                        for (size_t jj = 0; jj < J; jj += C_blocks * DIM) {
                            uint32_t norm_C_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + jj + r;
                            if (jj + C_blocks*DIM >= J) {
                                norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                            } else {
                                norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                            }
                            void * const C_dram_addr = (int8_t*)out +
                                (i*C_row_stride + jj) * sizeof(elem_t) +
                                r * C_row_stride * sizeof(elem_t);
                            const size_t cols = J - jj < C_blocks * DIM ? J - jj : C_blocks * DIM;
                            gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                        }
                    }
                }
            }
        }

    }
#else
    printf("Normalizations not supported in this Gemmini config\n");
    exit(1);
#endif
}

static void tiled_norm(const size_t I, const size_t J,
        const size_t tile_I, const size_t tile_J,
        const acc_t * in,
        elem_t * out,
        const acc_scale_t C_scale,
        int act,
        enum tiled_matmul_type_t norm_type) {

    gemmini_extended_config_st(J * sizeof(elem_t), act & 3, C_scale);
    gemmini_config_ex(WS, 0, 0); // TODO is this actually required?

    gemmini_extended4_config_ld(J * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, DIM, 0);
    gemmini_extended4_config_ld(J * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, DIM, 1);

    if (act == SOFTMAX) {
        const scale_t a = 0.3585;
        const scale_t b = 1.353;
        const scale_t c = 0.344;

        // TODO let bert-scale be set by the programmer
        acc_scale_t bert_scale = 0.05;
        const acc_t qln2 = (int) (0.693147 / bert_scale);
        const acc_t qln2_inv = 65536 / qln2;
        const acc_t qb = b / bert_scale;
        const acc_t qc = c / (a*bert_scale*bert_scale);

        gemmini_config_norm(qln2, 0, 0, 1, 0, qb, qc);
        gemmini_config_norm(qln2_inv, 1, 0, 1, 0, qb, qc);
    }

    for (size_t i = 0; i < I; i += tile_I) {
        for (size_t j = 0; j < J; j += tile_J) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

            const acc_t * in_ = in + i * J + j;
            elem_t * out_ = out + i * J + j;

            sp_tiled_norm(I_tile, J_tile,
                    in_, out_,
                    J, J,
                    act);
        }
    }

    gemmini_fence();
}

void tiled_norm_auto(const size_t I, const size_t J,
        const acc_t * in,
        elem_t * out,
        const acc_scale_t C_scale,
        int act,
        enum tiled_matmul_type_t norm_type) {

    size_t tile_I = I, tile_J = J;
    size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

    while (total_acc_rows > ACC_ROWS) {
        if (tile_I > 1) {
            tile_I--;
        } else {
            // TODO we should be able to tile over J as well to avoid this issue
            printf("Can't fit pre-normalized tensor into accumulator");
            exit(1);
        }

        total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    }

    if (norm_type) {
      tiled_norm(I, J, tile_I, tile_J,
            in, out,
            C_scale, act, norm_type);
    } else {
      printf("Unsupported type\n");
      exit(1);
    }
}

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ qkv_b, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out_buf_acc = out_buf * Wo
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out_buf_acc = FF2(out)
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b,
        const acc_t * Wo_cross_b,

        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,

        Wq_b, Wk_b, Wv_b,
        Wo_b,

        Q_buf, K_buf, V_buf,
        attn_buf, out_buf, out_buf_acc);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,

            Wq_cross_b, Wk_cross_b, Wv_cross_b,
            Wo_cross_b,

            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf, out_buf_acc);

    uint64_t end = read_cycles();

    return end - start;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static const acc_t Wqkvo_b[4][hidden_dim]; \
    static const acc_t Wqkvo_cross_b[4][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[expansion_dim]; \
    static const acc_t ff2_b[hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static acc_t out_buf_acc[seq_len][hidden_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            compression_factor, \
            \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            \
            Wqkvo_b[0], Wqkvo_b[1], Wqkvo_b[2], \
            Wqkvo_b[2], \
            Wqkvo_cross_b[0], Wqkvo_cross_b[1], Wqkvo_cross_b[2], \
            Wqkvo_cross_b[3], \
            \
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, out_buf_acc, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    // Buffers for layer inputs/outputs
    static elem_t layer_input[512][768];
    static elem_t layer_output[512][768];
    
    // Initialize first layer input
    // (in practice this would be token embeddings + positional encodings)
    
    // Run 12 encoder layers sequentially, chaining outputs to inputs
    for (int layer = 0; layer < LAYERS; layer++) {
        const elem_t* current_input = (layer == 0) ? layer_input : layer_output;
        
        uint64_t cycles = ENCODER_DECODER(768, 3072, 12, 12, 512, 1, 
                                        current_input, NULL, layer_output);
    }

    exit(0);
}
