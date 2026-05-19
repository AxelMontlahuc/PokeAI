#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stdint.h>
#include "agent.h"
#include "adam.h"

typedef struct CheckpointHeader {
    uint32_t magic;
    int epoch;
} CheckpointHeader;

int save_checkpoint(char* filepath, Agent* agent, Optimizer* optim, int epoch);
int load_checkpoint(char* filepath, Agent* agent, Optimizer* optim, int* epoch);

#endif
