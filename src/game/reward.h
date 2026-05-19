#ifndef REWARD_H
#define REWARD_H

#include <stdbool.h>

#include "config.h"

double reward(int old_state[INPUT_SIZE], int new_state[INPUT_SIZE]);
void reset_flags();
bool is_done(void);

#endif