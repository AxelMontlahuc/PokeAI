#ifndef LIBRETRO_EMU_H
#define LIBRETRO_EMU_H

#include <stdint.h>
#include <stddef.h>

#include "config.h"

int gba_create(const char* core_so, const char* rom_path);
void gba_destroy();
long gba_ram(uint32_t addr, size_t nbytes);
int gba_button(int button_code);
int gba_reset(const char* savestate);
void gba_savestate(const char* save_path);
void gba_run(int frames);
void gba_screen(const char* path);
void gba_state(int state[INPUT_SIZE]);
void gba_set_core_silent(int silent);
void gba_set_press_timing(int hold_frames, int total_frames);
void gba_set_video_enabled(int enabled);

#endif