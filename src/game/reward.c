#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "config.h"
#include "reward.h"

bool PLAYER_HOUSE_FLAG = false;
bool PLAYER_ROOM_FLAG = false;
bool CLOCK_FLAG = false;            // Pour l'instant, pas d'adresse mémoire assez fiable
bool PLAYER_HOUSE_2_FLAG = false;
bool LITTLEROOT_FLAG = false;
bool RIVAL_HOUSE_FLAG = false;
bool RIVAL_ROOM_FLAG = false;
bool RIVAL_HOUSE_2_FLAG = false;
bool LITTLEROOT_2_FLAG = false;
bool ROUTE_101_FLAG = false;
bool LABORATORY_FLAG = false;
bool LITTLEROOT_3_FLAG = false;
bool ROUTE_101_2_FLAG = false;
bool OLDALE_FLAG = false;
bool ROUTE_103_FLAG = false;      // Pas encore implémenté à partir d'ici
bool LABORATORY_2_FLAG = false;
bool ROUTE_102_FLAG = false;
bool PETALBURG_FLAG = false;
bool ROUTE_104_FLAG = false;
bool PETALBURG_WOODS_FLAG = false;
bool RUSTBORO_FLAG = false;

void reset_flags() {
    PLAYER_HOUSE_FLAG = false;
    PLAYER_ROOM_FLAG = false;
    CLOCK_FLAG = false;
    PLAYER_HOUSE_2_FLAG = false;
    LITTLEROOT_FLAG = false;
    RIVAL_HOUSE_FLAG = false;
    RIVAL_ROOM_FLAG = false;
    RIVAL_HOUSE_2_FLAG = false;
    LITTLEROOT_2_FLAG = false;
    ROUTE_101_FLAG = false;
    LABORATORY_FLAG = false;
    LITTLEROOT_3_FLAG = false;
    ROUTE_101_2_FLAG = false;
    OLDALE_FLAG = false;
    ROUTE_103_FLAG = false;
    LABORATORY_2_FLAG = false;
    ROUTE_102_FLAG = false;
    PETALBURG_FLAG = false;
    ROUTE_104_FLAG = false;
    PETALBURG_WOODS_FLAG = false;
    RUSTBORO_FLAG = false;
}

double reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]) {
	double r = -0.001;

	if (!PLAYER_HOUSE_FLAG && new_state[0] == 1) {
		r += 1;
		PLAYER_HOUSE_FLAG = true;
	}

    if (!PLAYER_ROOM_FLAG && new_state[0] == 257) {
        PLAYER_ROOM_FLAG = true;
        r += 1.7;
    }

    if (PLAYER_ROOM_FLAG && !PLAYER_HOUSE_2_FLAG && new_state[0] == 1) {
        PLAYER_HOUSE_2_FLAG = true;
        r += 1.2;
    }

    if (PLAYER_HOUSE_2_FLAG && !LITTLEROOT_FLAG && new_state[0] == 2304) {
        LITTLEROOT_FLAG = true;
        r += 1.7;
    }

    if (!RIVAL_HOUSE_FLAG && new_state[0] == 513) {
        RIVAL_HOUSE_FLAG = true;
        r += 3.0;
    }

    if (!RIVAL_ROOM_FLAG && new_state[0] == 769) {
        RIVAL_ROOM_FLAG = true;
        r += 1.6;
    }

    if (RIVAL_ROOM_FLAG && !RIVAL_HOUSE_2_FLAG && new_state[0] == 513) {
        RIVAL_HOUSE_2_FLAG = true;
        r += 1.6;
    }

    if (RIVAL_HOUSE_2_FLAG && !LITTLEROOT_2_FLAG && new_state[0] == 2304) {
        LITTLEROOT_2_FLAG = true;
        r += 1.8;
    }

    if (!ROUTE_101_FLAG && new_state[0] == 4096) {
        ROUTE_101_FLAG = true;
        r += 3.2;
    }

    if (ROUTE_101_FLAG && !LABORATORY_FLAG && new_state[0] == 1025) {
        LABORATORY_FLAG = true;
        r += 2.3;
    }

    if (LABORATORY_FLAG && !LITTLEROOT_3_FLAG && new_state[0] == 2304) {
        LITTLEROOT_3_FLAG = true;
        r += 1.8;
    }

    if (LITTLEROOT_3_FLAG && !ROUTE_101_2_FLAG && new_state[0] == 4096) {
        ROUTE_101_2_FLAG = true;
        r += 2.5;
    }

    if (!OLDALE_FLAG && new_state[0] == 2560) {
        OLDALE_FLAG = true;
        r += 4.0;
    }

    return r;
}

bool is_done() {
    return false;
}