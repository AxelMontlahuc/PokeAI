#ifndef REWARD_H
#define REWARD_H

#include <stdbool.h>

#include "config.h"

// Strcuture pour stocker les tiles explorées
typedef struct {
    int map;
    int x;
    int y;
} ExploredTile;

float reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]);
void reset_flags(void);
bool is_done(void);

#endif