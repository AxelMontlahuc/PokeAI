#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "config.h"
#include "reward.h"

bool PLAYER_HOUSE_FLAG = false;
bool PLAYER_ROOM_FLAG = false;
bool CLOCK_FLAG = false;
bool LITTLEROOT_FLAG = false;
bool RIVAL_HOUSE_FLAG = false;
bool RIVAL_ROOM_FLAG = false;
bool ROUTE_101_FLAG = false;
bool LABORATORY_FLAG = false;
bool OLDALE_FLAG = false;
bool ROUTE_103_FLAG = false;
bool LABORATORY_2_FLAG = false;
bool ROUTE_102_FLAG = false;
bool PETALBURG_FLAG = false;
bool ROUTE_104_FLAG = false;
bool PETALBURG_WOODS_FLAG = false;
bool RUSTBORO_FLAG = false;

double reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]) {
	double r = 0;

	if (!PLAYER_HOUSE_FLAG && new_state[0] == 1) {
		r += 1;
		PLAYER_HOUSE_FLAG = true;
	}

    if (!PLAYER_ROOM_FLAG && new_state[0] == 257) {
        PLAYER_ROOM_FLAG = true;
        r += 1.7;
    }

    if (PLAYER_ROOM_FLAG && !LITTLEROOT_FLAG && new_state[0] == 2304) {
        LITTLEROOT_FLAG = true;
        r += 3.5;
    }

    if (!RIVAL_HOUSE_FLAG && new_state[0] == 513) {
        RIVAL_HOUSE_FLAG = true;
        r += 2.2;
    }

    if (!RIVAL_ROOM_FLAG && RIVAL_HOUSE_FLAG && old_state[0] == 513 && new_state[0] == 2304) {
        r -= 3.0;
    }

    return r;
}

bool is_done() {
    return false;
}