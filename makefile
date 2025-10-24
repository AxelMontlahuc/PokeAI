CC := gcc
AR := ar
CSTD := -std=c99
CFLAGS := $(CSTD) -Wall -Wextra -O2 -Isrc -ImGBA-interface/include

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows
else
	UNAME_S := $(shell uname -s)
endif

ifeq ($(UNAME_S),Linux)
	LDFLAGS := -lm
	EXE :=
	MKDIR_P := mkdir -p
	RM_RF := rm -rf
else ifeq ($(UNAME_S),Darwin)
	# macOS
	LDFLAGS := -lm
	EXE :=
	MKDIR_P := mkdir -p
	RM_RF := rm -rf
else
	# Windows
	LDFLAGS := -lws2_32 -lm
	EXE := .exe
	MKDIR_P := powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path"
	RM_RF := cmd /C rmdir /S /Q
endif

BUILD_DIR := build
BIN_DIR := bin

COMMON_SRC := \
	src/func.c \
	src/policy.c \
	src/struct.c \
	src/checkpoint.c \
	src/reward.c \
	src/serializer.c

WORKER_SRC := \
	src/worker.c \
	src/state.c \
	$(COMMON_SRC)

LEARNER_SRC := \
	src/learner.c \
	src/state.c \
	$(COMMON_SRC)

WORKER_OBJS := $(WORKER_SRC:%.c=$(BUILD_DIR)/%.o)
LEARNER_OBJS := $(LEARNER_SRC:%.c=$(BUILD_DIR)/%.o)

MGBA_SRC := \
  mGBA-interface/src/mgba_connection.c \
  mGBA-interface/src/mgba_controller.c \
  mGBA-interface/src/mgba_map.c \
  mGBA-interface/src/mgba_intel.c

MGBA_OBJS := $(MGBA_SRC:%.c=$(BUILD_DIR)/%.o)
LIBMGBA := $(BUILD_DIR)/libmgba_controller.a

WORKER_TARGET := $(BIN_DIR)/worker$(EXE)
LEARNER_TARGET := $(BIN_DIR)/learner$(EXE)

.PHONY: all clean run dirs

all: $(WORKER_TARGET) $(LEARNER_TARGET)

dirs:
ifneq ($(UNAME_S),Windows)
	@$(MKDIR_P) $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR)/src $(BUILD_DIR)/mGBA-interface $(BUILD_DIR)/mGBA-interface/src
else
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BIN_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/src' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mGBA-interface' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mGBA-interface/src' | Out-Null"
endif

$(WORKER_TARGET): dirs $(WORKER_OBJS) $(LIBMGBA)
	$(CC) $(WORKER_OBJS) -o $@ $(LIBMGBA) $(LDFLAGS)

$(LEARNER_TARGET): dirs $(LEARNER_OBJS) $(LIBMGBA)
	$(CC) $(LEARNER_OBJS) -o $@ $(LIBMGBA) $(LDFLAGS)

$(LIBMGBA): dirs $(MGBA_OBJS)
	$(AR) rcs $@ $(filter %.o,$^)

$(BUILD_DIR)/%.o: %.c | dirs
	$(CC) $(CFLAGS) -c $< -o $@

run: $(LEARNER_TARGET)
	$(LEARNER_TARGET)

clean:
	-@echo Cleaning...
ifneq ($(UNAME_S),Windows)
	-@$(RM_RF) $(BUILD_DIR) $(BIN_DIR)
else
	-@cmd /C rmdir /S /Q $(BUILD_DIR) $(BIN_DIR) 2> NUL || exit 0
endif
