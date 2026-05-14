CC := gcc

TARGET := ai

SRC := \
	src/nn/agent.c \
	src/nn/lstm.c \
	src/nn/dense.c \
	src/nn/ppo.c \
	src/nn/adam.c \
	src/emu/libretro_emu.c \
	src/game/reward.c

INCLUDES := -Isrc -Isrc/nn -Isrc/emu -Isrc/game

COMMON_CFLAGS := -std=c99 -Wall -Wextra -Wshadow -Wconversion -Wpedantic \
	-include stdbool.h $(INCLUDES)

DEBUG_CFLAGS := -O0 -g3 -DDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer
FAST_CFLAGS := -Ofast -flto -march=native -funroll-loops -fno-math-errno -funsafe-math-optimizations -DNDEBUG

LDFLAGS := -lm -ldl

.PHONY: all debug fast clean

all: debug

debug:
	$(CC) $(COMMON_CFLAGS) $(DEBUG_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

fast:
	$(CC) $(COMMON_CFLAGS) $(FAST_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
