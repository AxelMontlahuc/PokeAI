#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "reward.h"
#include "libretro_emu.h"

ExploredTile visited_tiles[TRAJ_SIZE];
int visited_tiles_count = 0;

typedef struct {
    uint16_t var_id;
    uint16_t rewarded_states;
} InternalStateVarReward;

enum {
    INTERNAL_STATE_VAR_COUNT = 5,
    INTERNAL_STATE_MAX_STATE = 7
};

#define INTERNAL_STATE_REWARD 1.5f
#define STATE_BIT(state) ((uint16_t)(1u << (state)))

static const InternalStateVarReward INTERNAL_STATE_VARS[INTERNAL_STATE_VAR_COUNT] = {
    {0x4084, STATE_BIT(2) | STATE_BIT(3) | STATE_BIT(4) | STATE_BIT(5)}, // VAR_BIRCH_LAB_STATE
    {0x4060, STATE_BIT(1) | STATE_BIT(2)},                              // VAR_ROUTE101_STATE
    {0x4050, STATE_BIT(1) | STATE_BIT(2) | STATE_BIT(4)},                // VAR_LITTLEROOT_TOWN_STATE
    {0x4092, STATE_BIT(3) | STATE_BIT(5) | STATE_BIT(6) | STATE_BIT(7)},  // VAR_LITTLEROOT_INTRO_STATE
    {0x408D, STATE_BIT(2) | STATE_BIT(3)},                               // VAR_LITTLEROOT_RIVAL_STATE
};

bool INTERNAL_STATE_VAR_STATES_SEEN[INTERNAL_STATE_VAR_COUNT][INTERNAL_STATE_MAX_STATE + 1] = {{false}};

void reset_flags() {
    memset(INTERNAL_STATE_VAR_STATES_SEEN, 0, sizeof(INTERNAL_STATE_VAR_STATES_SEEN));
    
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

float reward_internal_state_var(int index) {
    const InternalStateVarReward *var = &INTERNAL_STATE_VARS[index];
    int state = get_emerald_var(var->var_id);

    if (state <= 0 || state > INTERNAL_STATE_MAX_STATE) {
        return 0.0f;
    }
    if ((var->rewarded_states & STATE_BIT(state)) == 0) {
        return 0.0f;
    }
    if (INTERNAL_STATE_VAR_STATES_SEEN[index][state]) {
        return 0.0f;
    }

    INTERNAL_STATE_VAR_STATES_SEEN[index][state] = true;
    return INTERNAL_STATE_REWARD;
}

float reward_internal_state_vars(void) {
    float r = 0.0f;

    for (int i = 0; i < INTERNAL_STATE_VAR_COUNT; i++) {
        r += reward_internal_state_var(i);
    }

    return r;
}

float reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]) {
    float r = STEP_PENALTY;

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
        r += (float)level_diff * WEIGHT_LEVEL_UP;
    }

    // --- 3. Récompense de progression interne
    r += reward_internal_state_vars();

    return r;
}

bool is_done() {
    return false;
}
