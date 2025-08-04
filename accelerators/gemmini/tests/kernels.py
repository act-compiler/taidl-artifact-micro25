import time
import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

# Default to sim_16 for most kernels
target_dir = os.path.join(os.path.dirname(base_dir), "sim_16")
sys.path.append(target_dir)

from decorator import kernel
import api as api

# Cache for loaded modules to avoid reimporting
_module_cache = {}


def get_gemmini_modules(DIM):
    """Dynamically import modules from the appropriate sim directory based on DIM"""
    import importlib.util
    import sys

    cache_key = f"DIM_{DIM}"
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    sim_dir = os.path.join(os.path.dirname(base_dir), f"sim_{DIM}")
    api_path = os.path.join(sim_dir, 'api.py')
    spec_api = importlib.util.spec_from_file_location(f"api_{DIM}", api_path)
    api_module = importlib.util.module_from_spec(spec_api)
    spec_api.loader.exec_module(api_module)

    decorator_path = os.path.join(sim_dir, 'decorator.py')
    spec_decorator = importlib.util.spec_from_file_location(f"decorator_{DIM}", decorator_path)
    decorator_module = importlib.util.module_from_spec(spec_decorator)

    decorator_module.api = api_module
    spec_decorator.loader.exec_module(decorator_module)
    decorator_module.api = api_module

    decorator_module.set_simulation_backend("CPU")

    result = (decorator_module.kernel, api_module)
    _module_cache[cache_key] = result

    return result


def matmul_6():
    res = 0
    a_addr = 0
    b_addr = 12544 * 256
    output_addr = b_addr + 256 * 64
    zero_addr = output_addr + 12544 * 64 * 4
    A_SP = 0
    B_SP = (4096 / 256 * 7 + 4 * 3) * 16

    @kernel(hbm=zero_addr + 2048,
            input=[
                {'addr': a_addr, 'shape': (12544, 256), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (256, 64), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (12544, 64), 'dtype': np.int32},
            ]
            )
    def matmul_6_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("io", 0, 98)
        api.start_loop("i", 0, 8)
        for l in range(4):
            api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=f"{64} * %i + {res + l * 16}")
        for ko in range(4):
            api.mvin_spad(
                tiles=4,
                stride=256,
                hbm_addr=f"{16*256} * %i + {128 *256} * %io + {64*ko + a_addr}",
                sp_addr=f"{4096/256 * 16} * %i + {4 * 16 * ko + A_SP}")
            for g in range(4):
                api.mvin_spad(tiles=4, stride=64, hbm_addr=64 * 64 * ko + 16 * 64 * g + b_addr,
                              sp_addr=16 * ko * 16 + B_SP + 4 * 16 * g)
            for h in range(16):
                api.matmul_preload(
                    rs1=4096 / 256 * 16 * ko + h * 16 + B_SP,
                    rs2=f"{4 * 16} * %i + {(h % 4) * 16 + res}")
                api.matmul32_compute_accumulated(
                    rs1=f"{4096/256 * 16} * %i + {4 * ko * 16 + int(h/4) * 16 + A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=64,
                    hbm_addr=f"{16*64*4} * %i + {64*128*4} * %io + {output_addr + p * 64}",
                    acc_addr=f"{4 * 16 } * %i + {res + p * 16}")

        api.end_loop()
        api.end_loop()
    return matmul_6_


def matmul_4():
    res = 0
    a_addr = 0
    b_addr = 12544 * 64
    output_addr = b_addr + 256 * 64
    zero_addr = output_addr + 12544 * 256 * 4
    # print(zero_addr)
    A_SP = 0
    B_SP = 4 * 195 * 16

    @kernel(hbm=zero_addr + 2048,
            input=[
                {'addr': a_addr, 'shape': (12544, 64), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (64, 256), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (12544, 256), 'dtype': np.int32},
            ]
            )
    def matmul_4_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("io", 0, 4)
        api.start_loop("i", 0, 196)
        for j in range(4):

            for l in range(4):
                api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=(res + j * 4 * 16 + l * 16))

            api.mvin_spad(
                tiles=4,
                stride=64,
                hbm_addr=f"{16*64} * %i + {3136 *64} * %io + {a_addr}",
                sp_addr=f"{1024/256 * 16} * %i + {A_SP}")

            for g in range(4):
                api.mvin_spad(tiles=4, stride=256, hbm_addr=64 * j + 16 * 256 * g + b_addr,
                              sp_addr=16 * 16 * j + B_SP + 4 * 16 * g)
            for h in range(16):
                api.matmul_preload(rs1=4096 / 16 * j + h * 16 + B_SP,
                                   rs2=4 * 16 * j + (h % 4) * 16 + res)
                api.matmul32_compute_accumulated(
                    rs1=f"{1024/256*16} * %i + {int(h/4) * 16 + A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=256,
                    hbm_addr=f"{16*256*4} * %i + {256*3136*4} * %io + {64*4 * j + output_addr + p * 64}",
                    acc_addr=4 * j * 16 + res + p * 16)

        api.end_loop()
        api.end_loop()
    return matmul_4_


def matmul_27():
    res = 0
    a_addr = 0
    b_addr = 784 * 256
    output_addr = b_addr + 256 * 1024
    zero_addr = output_addr + 784 * 1024 * 4
    A_SP = 0
    B_SP = (4096 / 256 * 6 + 4 * 3) * 16

    @kernel(hbm=zero_addr + 4096,
            input=[
                {'addr': a_addr, 'shape': (784, 256), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (256, 1024), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (784, 1024), 'dtype': np.int32},
            ]
            )
    def matmul_27_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("io", 0, 7)
        api.start_loop("jo", 0, 2)
        api.start_loop("i", 0, 7)
        for j in range(8):

            for l in range(4):
                api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=(res + j * 4 * 16 + l * 16))

            for ko in range(4):
                api.mvin_spad(
                    tiles=4,
                    stride=256,
                    hbm_addr=f"{16*256} * %i + {112 *256} * %io + {64 * ko + a_addr}",
                    sp_addr=f"{4096/16} * %i + {64 * ko + A_SP}")
                for g in range(4):
                    api.mvin_spad(
                        tiles=4,
                        stride=1024,
                        hbm_addr=f"{512} * %jo + {64*1024 * ko + 64 * j + 16 * 1024 * g + b_addr}",
                        sp_addr=16384 /
                        16 *
                        j +
                        16 *
                        16 *
                        ko +
                        B_SP +
                        4 *
                        16 *
                        g)
                for h in range(16):
                    api.matmul_preload(rs1=16384 / 16 * j + 16 * 16 * ko + h *
                                       16 + B_SP, rs2=16 * 4 * j + (h % 4) * 16 + res)
                    api.matmul32_compute_accumulated(
                        rs1=f"{4096/16} * %i + {4 * 16 * ko + int(h/4) * 16 + A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=1024,
                    hbm_addr=f"{16*1024*4} * %i + {112*1024*4} * %io + {512*4} * %jo + {64*4 * j + output_addr + p * 64}",
                    acc_addr=4 *
                    16 *
                    j +
                    res +
                    p *
                    16)

        api.end_loop()
        api.end_loop()
        api.end_loop()
    return matmul_27_


def matmul_14():
    res = 0
    a_addr = 0
    b_addr = 3136 * 128
    output_addr = b_addr + 128 * 512
    zero_addr = output_addr + 3136 * 512 * 4
    A_SP = 0
    B_SP = (2048 / 256 * 48 + 4 * 1) * 16

    @kernel(hbm=zero_addr + 2048,
            input=[
                {'addr': a_addr, 'shape': (3136, 128), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (128, 512), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (3136, 512), 'dtype': np.int32},
            ]
            )
    def matmul_14_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("io", 0, 4)
        api.start_loop("i", 0, 49)
        for j in range(8):

            for l in range(4):
                api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=(res + j * 4 * 16 + l * 16))

            for ko in range(2):
                api.mvin_spad(
                    tiles=4,
                    stride=128,
                    hbm_addr=f"{16*128} * %i + {784 *128} * %io + {64 * ko + a_addr}",
                    sp_addr=f"{2048/16} * %i + {4 * 16 * ko + A_SP}")
                for g in range(4):
                    api.mvin_spad(
                        tiles=4,
                        stride=512,
                        hbm_addr=64 *
                        512 *
                        ko +
                        16 *
                        512 *
                        g +
                        64 *
                        j +
                        b_addr,
                        sp_addr=8192 /
                        16 *
                        j +
                        16 *
                        16 *
                        ko +
                        B_SP +
                        4 *
                        16 *
                        g)
                for h in range(16):
                    api.matmul_preload(rs1=8192 / 16 * j + 16 * 16 * ko + h *
                                       16 + B_SP, rs2=4 * j * 16 + (h % 4) * 16 + res)
                    api.matmul32_compute_accumulated(
                        rs1=f"{2048/16} * %i + {4 * 16 *  ko + int(h/4) * 16+ A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=512,
                    hbm_addr=f"{16*512*4} * %i + {784*512*4} * %io + {64*4 * j + output_addr + p * 64}",
                    acc_addr=4 * j * 16 + res + p * 16)

        api.end_loop()
        api.end_loop()
    return matmul_14_


def matmul_16():
    res = 0
    a_addr = 0
    b_addr = 1605632
    output_addr = 1671168
    zero_addr = output_addr + 3136 * 128 * 4
    A_SP = 0
    B_SP = (8192 / 256 * 13 + 4 * 7) * 16

    @kernel(hbm=zero_addr + 2048,
            input=[
                {'addr': a_addr, 'shape': (3136, 512), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (512, 128), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (3136, 128), 'dtype': np.int32},
            ]
            )
    def matmul_16_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("io", 0, 14)
        api.start_loop("i", 0, 14)
        for j in range(2):

            for l in range(4):
                api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=(res + j * 4 * 16 + l * 16))

            for ko in range(8):
                api.mvin_spad(
                    tiles=4,
                    stride=512,
                    hbm_addr=f"{512*16} * %i + {224 *512} * %io + {64 * ko + a_addr}",
                    sp_addr=f"{8192/16} * %i + {4 * 16 * ko + A_SP}")
                for g in range(4):
                    api.mvin_spad(
                        tiles=4,
                        stride=128,
                        hbm_addr=64 *
                        ko *
                        128 +
                        16 *
                        128 *
                        g +
                        64 *
                        j +
                        b_addr,
                        sp_addr=32768 /
                        16 *
                        j +
                        4096 /
                        16 *
                        ko +
                        B_SP +
                        4 *
                        16 *
                        g)
                for h in range(16):
                    api.matmul_preload(rs1=32768 / 16 * j + 16 * 16 * ko + h *
                                       16 + B_SP, rs2=4 * 16 * j + (h % 4) * 16 + res)
                    api.matmul32_compute_accumulated(
                        rs1=f"{8192/16} * %i + {4 * 16 * ko + int(h/4) * 16 + A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=128,
                    hbm_addr=f"{16*128*4} * %i + {224*128*4} * %io + {64*4 * j + output_addr + p * 64}",
                    acc_addr=4 * 16 * j + res + p * 16)

        api.end_loop()
        api.end_loop()
    return matmul_16_


def matmul_512():
    res = 0

    a_addr = 0
    b_addr = 262144
    output_addr = 524288
    zero_addr = 524288 + 262144 * 4

    A_SP = 0
    B_SP = (256 * 1 + 32 * 7 + 4 * 3) * 16

    @kernel(hbm=zero_addr + 4 * 512,
            input=[
                {'addr': a_addr, 'shape': (512, 512), 'dtype': np.int8},
                {'addr': b_addr, 'shape': (512, 512), 'dtype': np.int8},
            ],
            output=[
                {'addr': output_addr, 'shape': (512, 512), 'dtype': np.int32},
            ]
            )
    def matmul_512_() -> None:
        api.dataflow_config(dataflow=1)
        api.start_loop("ioo", 0, 2)
        api.start_loop("jo", 0, 2)
        api.start_loop("io", 0, 2)
        api.start_loop("i", 0, 8)
        for ji in range(4):

            for l in range(4):
                api.mvin_acc(rows=1, hbm_addr=zero_addr, acc_addr=(res + ji * 16 * 4 + l * 16))

            for ko in range(8):
                api.mvin_spad(
                    tiles=4,
                    stride=512,
                    hbm_addr=f"{512*16} * %i + {128 *512} * %io + {256*512} * %ioo + {64 * ko + a_addr}",
                    sp_addr=f"{256 * 16} * %io + {32 * 16} * %i + {4 * 16 * ko + A_SP}")
                for g in range(4):
                    api.mvin_spad(
                        tiles=4,
                        stride=512,
                        hbm_addr=f"256 * %jo + {64 * ji + 64 *512 * ko + b_addr + 16 * 512 * g}",
                        sp_addr=128 *
                        16 *
                        ji +
                        16 *
                        16 *
                        ko +
                        B_SP +
                        4 *
                        16 *
                        g)
                    for h in range(4):
                        api.matmul_preload(
                            rs1=128 * 16 * ji + 16 * 16 * ko + B_SP + 4 * 16 * g + 16 * h,
                            rs2=4 * 16 * ji + 16 * h + res)
                        api.matmul32_compute_accumulated(
                            rs1=f"{256 * 16} * %io + {32 * 16} * %i + {4 * 16 * ko + 16 * g + A_SP}", rs2=-1)

            for p in range(4):
                api.mvout_acc_stride(
                    stride=512,
                    hbm_addr=f"{8192*4} * %i + {65536*4} * %io + {131072*4} * %ioo + {256*4} * %jo + {64*4 * ji + output_addr + p * 64}",
                    acc_addr=4 *
                    16 *
                    ji +
                    res +
                    p *
                    16)

        api.end_loop()
        api.end_loop()
        api.end_loop()
        api.end_loop()
    return matmul_512_


def i_bert():
    SEQ_LEN = 512
    HIDDEN_DIM = 768
    FFN_DIM = 3072
    NUM_LAYERS = 12

    a_addr = 0
    wq_addr = SEQ_LEN * HIDDEN_DIM
    wk_addr = wq_addr + HIDDEN_DIM * HIDDEN_DIM * NUM_LAYERS
    wv_addr = wk_addr + HIDDEN_DIM * HIDDEN_DIM * NUM_LAYERS
    wo_addr = wv_addr + HIDDEN_DIM * HIDDEN_DIM * NUM_LAYERS
    w1_addr = wo_addr + HIDDEN_DIM * HIDDEN_DIM * NUM_LAYERS
    w2_addr = w1_addr + FFN_DIM * HIDDEN_DIM * NUM_LAYERS

    output_addr = w2_addr + FFN_DIM * HIDDEN_DIM * NUM_LAYERS

    q_out_addr = output_addr
    k_out_addr = q_out_addr + SEQ_LEN * HIDDEN_DIM
    v_out_addr = k_out_addr + SEQ_LEN * HIDDEN_DIM
    qk_addr = v_out_addr + SEQ_LEN * HIDDEN_DIM
    softmax_addr = qk_addr + SEQ_LEN * HIDDEN_DIM
    attn_out_addr = softmax_addr + SEQ_LEN * HIDDEN_DIM
    proj_out_addr = attn_out_addr + SEQ_LEN * HIDDEN_DIM
    ffn1_out_addr = proj_out_addr + SEQ_LEN * HIDDEN_DIM
    ffn_out_addr = ffn1_out_addr + SEQ_LEN * FFN_DIM
    zero_addr = ffn_out_addr + SEQ_LEN * HIDDEN_DIM

    A_SP = 0
    B_SP = 1
    OUT_SP = 2

    @kernel(hbm=zero_addr + 4 * 512,
            input=[
                {'addr': a_addr, 'shape': (SEQ_LEN, HIDDEN_DIM), 'dtype': np.int8},  # Input
                {'addr': wq_addr, 'shape': (NUM_LAYERS, HIDDEN_DIM,
                                            HIDDEN_DIM), 'dtype': np.int8},  # Wq
                {'addr': wk_addr, 'shape': (NUM_LAYERS, HIDDEN_DIM,
                                            HIDDEN_DIM), 'dtype': np.int8},  # Wk
                {'addr': wv_addr, 'shape': (NUM_LAYERS, HIDDEN_DIM,
                                            HIDDEN_DIM), 'dtype': np.int8},  # Wv
                {'addr': wo_addr, 'shape': (NUM_LAYERS, HIDDEN_DIM,
                                            HIDDEN_DIM), 'dtype': np.int8},  # Wo
                {'addr': w1_addr, 'shape': (NUM_LAYERS, HIDDEN_DIM, FFN_DIM),
                 'dtype': np.int8},  # W1
                {'addr': w2_addr, 'shape': (NUM_LAYERS, FFN_DIM, HIDDEN_DIM),
                 'dtype': np.int8},  # W2
            ],
            output=[
                {'addr': ffn_out_addr, 'shape': (SEQ_LEN, HIDDEN_DIM), 'dtype': np.int8},
            ]
            )
    def i_bert_() -> None:
        class counter:
            matmul_count = 0

        # Stride is probably wrong, but shouldn't change timing much
        def tiled_matmul(A_base, B_base, C_base, M, N, K, count):
            tile_N = N // 16
            tile_M = M // 16
            tile_K = K // 16
            api.start_loop(f"i.{count.matmul_count}", 0, tile_M)
            api.start_loop(f"j.{count.matmul_count}", 0, tile_N)
            api.mvin_spad(tiles=1, stride=16, hbm_addr=zero_addr, sp_addr=0)
            for k in range(tile_K):
                api.mvin_spad(
                    tiles=1,
                    stride=16,
                    hbm_addr=f"{K * 16} * %i.{count.matmul_count} + {16 * 16 * k} + {A_base}",
                    sp_addr=A_SP)
                api.mvin_spad(
                    tiles=1,
                    stride=16,
                    hbm_addr=f"{16 * 16} * %j.{count.matmul_count} + {N * 16 * k} + {B_base}",
                    sp_addr=B_SP)
                api.matmul_preload(rs1=B_SP, rs2=OUT_SP)
                api.matmul32_compute_accumulated_v2(rs1=A_SP, rs2=-1)
            api.mvout_spad(
                stride=16,
                rows=16,
                hbm_addr=f"{N * 16} * %i.{count.matmul_count} + {16 * 16} * %j.{count.matmul_count} + {C_base}",
                sp_addr=OUT_SP)
            api.end_loop()
            api.end_loop()
            count.matmul_count += 1
        matmul_count = counter()

        api.dataflow_config(dataflow=1)

        api.start_loop("layers", 0, 12)

        WQ_L = f"{HIDDEN_DIM*HIDDEN_DIM} * %layers + {wq_addr}"
        WK_L = f"{HIDDEN_DIM*HIDDEN_DIM} * %layers + {wk_addr}"
        WV_L = f"{HIDDEN_DIM*HIDDEN_DIM} * %layers + {wv_addr}"
        WO_L = f"{HIDDEN_DIM*HIDDEN_DIM} * %layers + {wo_addr}"
        W1_L = f"{HIDDEN_DIM*FFN_DIM} * %layers + {w1_addr}"
        W2_L = f"{FFN_DIM*HIDDEN_DIM} * %layers + {w2_addr}"

        # Q, K, V projections
        tiled_matmul(a_addr, WQ_L, q_out_addr, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM, matmul_count)
        tiled_matmul(a_addr, WK_L, k_out_addr, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM, matmul_count)
        tiled_matmul(a_addr, WV_L, v_out_addr, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM, matmul_count)

        # Q x K^T -> 128 x 128
        tiled_matmul(q_out_addr, k_out_addr, qk_addr, SEQ_LEN, HIDDEN_DIM, SEQ_LEN, matmul_count)

        # Softmax
        api.softmax(rs1=qk_addr, out_sp=softmax_addr, batch=SEQ_LEN, features=SEQ_LEN)

        # Softmax x V -> Attention Output
        tiled_matmul(
            softmax_addr,
            v_out_addr,
            attn_out_addr,
            SEQ_LEN,
            SEQ_LEN,
            HIDDEN_DIM,
            matmul_count)

        # Output projection
        tiled_matmul(
            attn_out_addr,
            WO_L,
            proj_out_addr,
            SEQ_LEN,
            HIDDEN_DIM,
            HIDDEN_DIM,
            matmul_count)

        # FFN Layer 1
        tiled_matmul(proj_out_addr, W1_L, ffn1_out_addr, SEQ_LEN, HIDDEN_DIM, FFN_DIM, matmul_count)

        # GELU
        api.gelu(rs1=ffn1_out_addr, out_sp=ffn1_out_addr, batch=SEQ_LEN, features=FFN_DIM)

        # FFN Layer 2
        tiled_matmul(ffn1_out_addr, W2_L, ffn_out_addr, SEQ_LEN, FFN_DIM, HIDDEN_DIM, matmul_count)

        api.end_loop()

    return i_bert_


def gemmini_kernel(DIM=16, iterations=16, df=0, xla_while=0):
    # Get the appropriate modules for this DIM
    kernel_decorator, api_module = get_gemmini_modules(DIM)

    hbm_size = (DIM * DIM) * (iterations * 3 + 1)
    a_addr = 0
    b_addr = iterations * DIM * DIM
    d_addr = (iterations + 1) * DIM * DIM

    c_addr = DIM * DIM * iterations * 2 + DIM * DIM

    @kernel_decorator(hbm=hbm_size,
                      input=[
                          {'addr': a_addr, 'shape': (DIM * iterations, DIM), 'dtype': np.int8},
                          {'addr': b_addr, 'shape': (DIM, DIM), 'dtype': np.int8},
                          {'addr': d_addr, 'shape': (DIM * iterations, DIM), 'dtype': np.int8},
                      ],
                      output=[
                          {'addr': c_addr, 'shape': (iterations * DIM, DIM), 'dtype': np.int8},
                      ]
                      )
    def gemmini_kernel() -> None:
        # A
        for i in range(0, iterations):
            api_module.mvin_spad(tiles=1, stride=DIM, hbm_addr=DIM * DIM * i, sp_addr=DIM * i)
        # B
        api_module.mvin_spad(
            tiles=1,
            stride=DIM,
            hbm_addr=DIM *
            DIM *
            iterations,
            sp_addr=DIM *
            iterations)
        # D
        for i in range(0, iterations):
            api_module.mvin_spad(tiles=1,
                                 stride=DIM,
                                 hbm_addr=DIM * DIM * i + (DIM * DIM * (iterations + 1)),
                                 sp_addr=DIM * i + (DIM * (iterations + 1)))

        api_module.dataflow_config(dataflow=df)
        if (df == 0):
            for i in range(iterations):
                api_module.matmul_preload(rs1=DIM * (iterations + 1 + i),
                                          rs2=DIM * (2 * iterations + 1 + i))
                api_module.matmul8_compute_preloaded(rs1=i * DIM, rs2=iterations * DIM)
        elif (df == 1):
            api_module.matmul_preload(rs1=DIM * (iterations), rs2=DIM * (2 * iterations + 1))
            api_module.matmul8_compute_preloaded(rs1=0, rs2=(iterations + 1) * DIM)
            for i in range(1, iterations):
                api_module.matmul_preload(rs1=0, rs2=DIM * (2 * iterations + 1 + i))
                api_module.matmul8_compute_accumulated(
                    rs1=DIM * (i), rs2=(iterations + i + 1) * DIM)
        for i in range(0, iterations):
            api_module.mvout_spad(stride=DIM, hbm_addr=(i + iterations * 2 + 1)
                                  * DIM * DIM, sp_addr=DIM * (i + iterations * 2 + 1), rows=DIM)

    @kernel_decorator(hbm=hbm_size,
                      input=[
                          {'addr': a_addr, 'shape': (DIM * iterations, DIM), 'dtype': np.int8},
                          {'addr': b_addr, 'shape': (DIM, DIM), 'dtype': np.int8},
                          {'addr': d_addr, 'shape': (DIM * iterations, DIM), 'dtype': np.int8},
                      ],
                      output=[
                          {'addr': c_addr, 'shape': (iterations * DIM, DIM), 'dtype': np.int8},
                      ]
                      )
    def gemmini_kernel_loop() -> None:
        api_module.start_loop("j", 0, iterations * 2 + 1)
        api_module.mvin_spad(
            tiles=1,
            stride=DIM,
            hbm_addr=f"{DIM * DIM} * %j + 0",
            sp_addr=f"{DIM} * %j + 0")
        api_module.end_loop()
        api_module.dataflow_config(dataflow=df)
        if (df == 0):
            api_module.start_loop("i", 0, iterations)
            api_module.matmul_preload(rs1=f"{DIM} * %i + {DIM * (iterations + 1)}",
                                      rs2=f"{DIM} * %i + {DIM * (2 * iterations + 1)}")
            api_module.matmul8_compute_preloaded(rs1=f"{DIM} * %i + 0", rs2=iterations * DIM)
            api_module.end_loop()
        elif (df == 1):
            api_module.matmul_preload(rs1=DIM * (iterations), rs2=DIM * (2 * iterations + 1))
            api_module.matmul8_compute_preloaded(rs1=0, rs2=(iterations + 1) * DIM)
            api_module.start_loop("i", 1, iterations)
            api_module.matmul_preload(rs1=0, rs2=f"{DIM} * %i + {DIM * (2 * iterations + 1)}")
            api_module.matmul8_compute_accumulated(
                rs1=f"{DIM} * %i + 0", rs2=f"{DIM} * %i + {(iterations + 1) * DIM}")
            api_module.end_loop()
        api_module.start_loop("k", 0, iterations)
        api_module.mvout_spad(
            stride=DIM,
            hbm_addr=f"{DIM * DIM} * %k + {(iterations * 2 + 1) * DIM * DIM}",
            sp_addr=f"{DIM} * %k + {(iterations * 2 + 1) * DIM}",
            rows=DIM)
        api_module.end_loop()

    if (xla_while):
        return gemmini_kernel_loop, hbm_size

    return gemmini_kernel, hbm_size
