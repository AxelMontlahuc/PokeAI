#ifndef MGBA_INTEL_H
#define MGBA_INTEL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mgba_connection.h"

int read_state(SOCKET sock, int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int bg0_out[32][32], int bg2_out[32][32]);
int read_behavior_map(SOCKET sock, int behavior_out[11][11], int* player_x, int* player_y);

#endif