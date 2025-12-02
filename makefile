CC := gcc
AR := ar
CSTD := -std=c99
CFLAGS := $(CSTD) -Wall -Wextra -O3 -march=native -ffast-math -Isrc -Igba -Imgba/include

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows
else
	UNAME_S := $(shell uname -s)
endif

ifeq ($(UNAME_S),Linux)
	LDFLAGS := -lm -ldl
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

MGBA_SRC := \
	mgba/src/mgba_connection.c \
	mgba/src/mgba_controller.c \
	mgba/src/mgba_intel.c
	
COMMON_SRC := \
	src/policy.c \
	src/constants.c \
	src/checkpoint.c \
	src/reward.c \
	src/serializer.c \
	gba/gba.c \
	$(MGBA_SRC)

WORKER_SRC := \
	src/worker.c \
	src/state.c \
	$(COMMON_SRC)

LEARNER_SRC := \
	src/learner.c \
	src/state.c \
	$(COMMON_SRC)

RUN_SRC := \
	src/run.c \
	src/state.c \
	$(COMMON_SRC)

# Savemaker binary
SAVEMAKER_SRC := \
	src/savemaker.c \
	src/state.c \
	$(COMMON_SRC)

WORKER_OBJS := $(WORKER_SRC:%.c=$(BUILD_DIR)/%.o)
LEARNER_OBJS := $(LEARNER_SRC:%.c=$(BUILD_DIR)/%.o)
RUN_OBJS := $(RUN_SRC:%.c=$(BUILD_DIR)/%.o)
SAVEMAKER_OBJS := $(SAVEMAKER_SRC:%.c=$(BUILD_DIR)/%.o)

WORKER_TARGET := $(BIN_DIR)/worker$(EXE)
LEARNER_TARGET := $(BIN_DIR)/learner$(EXE)
RUN_TARGET := $(BIN_DIR)/run$(EXE)
SAVEMAKER_TARGET := $(BIN_DIR)/savemaker$(EXE)


.PHONY: all clean run dirs

all: $(WORKER_TARGET) $(LEARNER_TARGET) $(RUN_TARGET) $(SAVEMAKER_TARGET)

dirs:
ifneq ($(UNAME_S),Windows)
	@$(MKDIR_P) $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR)/src $(BUILD_DIR)/gba $(BUILD_DIR)/mgba $(BUILD_DIR)/mgba/src
else
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BIN_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/src' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/gba' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mgba' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mgba/src' | Out-Null"
endif

$(WORKER_TARGET): dirs $(WORKER_OBJS)
	$(CC) $(WORKER_OBJS) -o $@ $(LDFLAGS)

$(LEARNER_TARGET): dirs $(LEARNER_OBJS)
	$(CC) $(LEARNER_OBJS) -o $@ $(LDFLAGS)

$(RUN_TARGET): dirs $(RUN_OBJS)
	$(CC) $(RUN_OBJS) -o $@ $(LDFLAGS)

$(SAVEMAKER_TARGET): dirs $(SAVEMAKER_OBJS)
	$(CC) $(SAVEMAKER_OBJS) -o $@ $(LDFLAGS)

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
