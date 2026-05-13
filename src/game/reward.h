#ifndef REWARD_H
#define REWARD_H

#include "config.h"

double reward(double old_state[INPUT_SIZE], double new_state[INPUT_SIZE]);
bool is_done(double state[INPUT_SIZE]);

#endif