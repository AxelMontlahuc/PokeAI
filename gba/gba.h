#ifndef PICOAPI_H
#define PICOAPI_H

#include <stdint.h>
#include <stddef.h>

int gba_create(const char* core_so, const char* rom_path);
void gba_destroy();
long gba_ram(uint32_t addr, size_t nbytes);
int gba_state(int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int bg0_out[32][32], int bg2_out[32][32]);
int gba_button(int button_code);
int gba_reset(const char* savestate);
void gba_savestate(const char* save_path);
void gba_run(int frames);
void gba_screen(const char* path);

#endif