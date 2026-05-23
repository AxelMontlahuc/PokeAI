CC := gcc

TARGET := ai

SRC := \
	src/nn/agent.c \
	src/nn/lstm.c \
	src/nn/dense.c \
	src/nn/ppo.c \
	src/nn/adam.c \
	src/emu/libretro_emu.c \
	src/game/reward.c \
	src/io/checkpoint.c

INCLUDES := -Isrc -Isrc/nn -Isrc/emu -Isrc/game -Isrc/io

COMMON_CFLAGS := -std=c99 -Wall -Wextra -Wshadow -Wconversion -Wpedantic \
	-include stdbool.h $(INCLUDES)

DEBUG_CFLAGS := -O0 -g3 -DDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer
FAST_CFLAGS := -Ofast -flto -march=native -funroll-loops -fno-math-errno -funsafe-math-optimizations -DNDEBUG

LDFLAGS := -lm -ldl

.PHONY: all debug fast fast32 clean record_run record_run32 gpsp_savestate bench_emu_buttons bench_emu_buttons32

all: debug

debug:
	$(CC) $(COMMON_CFLAGS) $(DEBUG_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

fast:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

fast32:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -m32 -o $(TARGET)32 $(SRC) $(LDFLAGS) -m32

record_run:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -o utils/record_run utils/record_run.c src/nn/lstm.c src/nn/dense.c src/nn/ppo.c src/nn/adam.c src/emu/libretro_emu.c src/game/reward.c src/io/checkpoint.c $(LDFLAGS)

record_run32:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -m32 -o utils/record_run32 utils/record_run.c src/nn/lstm.c src/nn/dense.c src/nn/ppo.c src/nn/adam.c src/emu/libretro_emu.c src/game/reward.c src/io/checkpoint.c $(LDFLAGS) -m32

gpsp_savestate:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -m32 -o utils/gpsp_savestate utils/gpsp_savestate.c src/emu/libretro_emu.c $(LDFLAGS) -m32

bench_emu_buttons:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -o utils/bench_emu_buttons utils/bench_emu_buttons.c src/emu/libretro_emu.c $(LDFLAGS)

bench_emu_buttons32:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -m32 -o utils/bench_emu_buttons32 utils/bench_emu_buttons.c src/emu/libretro_emu.c $(LDFLAGS) -m32

clean:
	rm -f $(TARGET) $(TARGET)32 utils/record_run utils/record_run32 utils/gpsp_savestate utils/bench_emu_buttons utils/bench_emu_buttons32
