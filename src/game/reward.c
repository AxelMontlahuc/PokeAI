#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#include "config.h"
#include "reward.h"

ExploredTile visited_tiles[TRAJ_SIZE];
int visited_tiles_count = 0;

// Flags
bool PLAYER_HOUSE_FLAG = false;
bool PLAYER_ROOM_FLAG = false;
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

    visited_tiles_count = 0;
}

bool is_tile_visited(int map, int x, int y) {
    for (int i = 0; i < visited_tiles_count; i++) {
        if (visited_tiles[i].map == map && visited_tiles[i].x == x && visited_tiles[i].y == y) {
            return true;
        }
    }
    return false;
}

void add_visited_tile(int map, int x, int y) {
    if (visited_tiles_count < TRAJ_SIZE) {
        visited_tiles[visited_tiles_count].map = map;
        visited_tiles[visited_tiles_count].x = x;
        visited_tiles[visited_tiles_count].y = y;
        visited_tiles_count++;
    }
}

int get_party_level_sum(int state[INPUT_SIZE]) {
    int level_sum = 0;
    for (int i = 0; i < 6; i++) {
        int level = state[3 + i * 3];
        if (level > 0 && level <= 100) {
            level_sum += level;
        }
    }
    return level_sum;
}

double reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]) {
    double r = STEP_PENALTY;

    // --- 1. Récompense d'exploration
    int new_map = new_state[0];
    int new_x = new_state[1];
    int new_y = new_state[2];

    if (!is_tile_visited(new_map, new_x, new_y)) {
        add_visited_tile(new_map, new_x, new_y);
        r += WEIGHT_EXPLORATION;
    }

    // --- 2. Récompense de combat
    int old_level_sum = get_party_level_sum(old_state);
    int new_level_sum = get_party_level_sum(new_state);
    
    if (new_level_sum > old_level_sum) {
        int level_diff = new_level_sum - old_level_sum;
        r += (double)level_diff * WEIGHT_LEVEL_UP;
    }

    // --- 3. Récompense de flags
    if (!PLAYER_HOUSE_FLAG && new_map == 1) {
        PLAYER_HOUSE_FLAG = true;
        r += 1;
    }
    
    if (!PLAYER_ROOM_FLAG && new_map == 257) {
        PLAYER_ROOM_FLAG = true;
        r += 1.7;
    }

    if (PLAYER_ROOM_FLAG && !PLAYER_HOUSE_2_FLAG && new_map == 1) {
        PLAYER_HOUSE_2_FLAG = true;
        r += 1.2;
    }

    if (PLAYER_HOUSE_2_FLAG && !LITTLEROOT_FLAG && new_map == 2304) {
        LITTLEROOT_FLAG = true;
        r += 1.7;
    }
    if (!RIVAL_HOUSE_FLAG && new_map == 513) {
        RIVAL_HOUSE_FLAG = true;
        r += 3.0;
    }
    if (!RIVAL_ROOM_FLAG && new_map == 769) {
        RIVAL_ROOM_FLAG = true;
        r += 1.6;
    }
    if (RIVAL_ROOM_FLAG && !RIVAL_HOUSE_2_FLAG && new_map == 513) {
        RIVAL_HOUSE_2_FLAG = true;
        r += 1.6;
    }
    if (RIVAL_HOUSE_2_FLAG && !LITTLEROOT_2_FLAG && new_map == 2304) {
        LITTLEROOT_2_FLAG = true;
        r += 1.8;
    }
    if (!ROUTE_101_FLAG && new_map == 4096) {
        ROUTE_101_FLAG = true;
        r += 3.2;
    }
    if (ROUTE_101_FLAG && !LABORATORY_FLAG && new_map == 1025) {
        LABORATORY_FLAG = true;
        r += 2.3;
    }
    if (LABORATORY_FLAG && !LITTLEROOT_3_FLAG && new_map == 2304) {
        LITTLEROOT_3_FLAG = true;
        r += 1.8;
    }
    if (LITTLEROOT_3_FLAG && !ROUTE_101_2_FLAG && new_map == 4096) {
        ROUTE_101_2_FLAG = true;
        r += 2.5;
    }
    if (!OLDALE_FLAG && new_map == 2560) {
        OLDALE_FLAG = true;
        r += 4.0;
    }

    return r;
}

bool is_done() {
    return false;
}