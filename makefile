CC := gcc
AR := ar
CSTD := -std=c99
CFLAGS := $(CSTD) -Wall -Wextra -O2 -Isrc -ImGBA-interface/include

# Detect platform for linker flags and file extensions
# Prefer environment variable OS on Windows to avoid calling uname
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

APP_SRC := \
  src/agent.c \
  src/func.c \
  src/policy.c \
  src/state.c \
  src/struct.c \
  src/checkpoint.c \
  src/reward.c

APP_OBJS := $(APP_SRC:%.c=$(BUILD_DIR)/%.o)

MGBA_SRC := \
  mGBA-interface/src/mgba_connection.c \
  mGBA-interface/src/mgba_controller.c \
  mGBA-interface/src/mgba_map.c \
  mGBA-interface/src/mgba_intel.c

MGBA_OBJS := $(MGBA_SRC:%.c=$(BUILD_DIR)/%.o)
LIBMGBA := $(BUILD_DIR)/libmgba_controller.a

TARGET := $(BIN_DIR)/agent$(EXE)

.PHONY: all clean run dirs

all: $(TARGET)

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

$(TARGET): dirs $(APP_OBJS) $(LIBMGBA)
	$(CC) $(APP_OBJS) -o $@ $(LIBMGBA) $(LDFLAGS)

$(LIBMGBA): dirs $(MGBA_OBJS)
	$(AR) rcs $@ $(filter %.o,$^)

$(BUILD_DIR)/%.o: %.c | dirs
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	$(TARGET)

clean:
	-@echo Cleaning...
ifneq ($(UNAME_S),Windows)
	-@$(RM_RF) $(BUILD_DIR) $(BIN_DIR)
else
	-@cmd /C rmdir /S /Q $(BUILD_DIR) $(BIN_DIR) 2> NUL || exit 0
endif
