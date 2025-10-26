#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "policy.h"

int saveLSTM(const char* path, const LSTM* net, uint64_t episode, uint64_t seed);
LSTM* loadLSTM(const char* path, uint64_t* episodes, uint64_t* seed);

#endif