#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "struct.h"

int saveLSTMCheckpoint(const char* path, const LSTM* net, uint64_t step, uint64_t rng_seed);
LSTM* loadLSTM(const char* path, uint64_t* episodes, uint64_t* rng_seed);

#endif