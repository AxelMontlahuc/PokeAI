CC := gcc
AR := ar
CSTD := -std=c99
CFLAGS := $(CSTD) -Wall -Wextra -O2 -Isrc -ImGBA-interface/include
LDFLAGS := -lws2_32 -lm

BUILD_DIR := build
BIN_DIR := bin

APP_SRC := \
  src/agent.c \
  src/func.c \
  src/policy.c \
  src/state.c \
  src/struct.c

APP_OBJS := $(APP_SRC:%.c=$(BUILD_DIR)/%.o)

MGBA_SRC := \
  mGBA-interface/src/mgba_connection.c \
  mGBA-interface/src/mgba_controller.c \
  mGBA-interface/src/mgba_map.c \
  mGBA-interface/src/mgba_intel.c

MGBA_OBJS := $(MGBA_SRC:%.c=$(BUILD_DIR)/%.o)
LIBMGBA := $(BUILD_DIR)/libmgba_controller.a

TARGET := $(BIN_DIR)/agent.exe

.PHONY: all clean run dirs

all: $(TARGET)

dirs:
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BIN_DIR)' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/src' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mGBA-interface' | Out-Null"
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/mGBA-interface/src' | Out-Null"

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
	-@rm -rf $(BUILD_DIR) $(BIN_DIR)
	-@rmdir /s /q $(BUILD_DIR) $(BIN_DIR) 2> NUL || exit 0