CFLAGS = -O1 -march=sapphirerapids
CC = gcc
BIN += cnn_inference_amx cnn_inference_mix
BIN += rnn_inference_amx
BIN += sgemm_avx
BIN += mem_format_prop_avx

all: $(BIN)

cnn_inference_amx: cnn_inference_amx.c
	$(CC) $(CFLAGS) $^ -o $@ -DNUM_BLOCKS=$(blocks) -DUNROLL=$(unroll)

rnn_inference_amx: rnn_inference_amx.c
	$(CC) $(CFLAGS) $^ -o $@ -DNUM_BLOCKS=$(blocks) -DUNROLL=$(unroll)

sgemm_avx: sgemm_avx.c
	$(CC) $(CFLAGS) $^ -o $@ -DNUM_BLOCKS=$(blocks) -DUNROLL=$(unroll)

mem_format_prop_avx: mem_format_prop_avx.c
	$(CC) $(CFLAGS) $^ -o $@ -DNUM_BLOCKS=$(blocks) -DUNROLL=$(unroll)

cnn_inference_mix: cnn_inference_mix.c
	$(CC) $(CFLAGS) $^ -o $@ -DNUM_BLOCKS=$(blocks) -DUNROLL=$(unroll)

clean:
	-rm $(BIN)

.PHONY: clean
