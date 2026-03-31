#ifndef LIBRETRO_EMU_H
#define LIBRETRO_EMU_H

#include <stdint.h>
#include <stddef.h>

int gba_create(const char* core_so, const char* rom_path);
void gba_destroy();
long gba_ram(uint32_t addr, size_t nbytes);
void gba_behavior_map(int behavior_out[11][11], int* player_x, int* player_y);
int gba_state(int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int behavior_out[11][11], int* player_x, int* player_y);
int gba_button(int button_code);
int gba_reset(const char* savestate);
void gba_savestate(const char* save_path);
void gba_run(int frames);
void gba_screen(const char* path);

#endif