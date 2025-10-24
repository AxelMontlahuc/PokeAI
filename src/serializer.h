#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "struct.h"

typedef struct {
    uint32_t magic;        // 'TRJ0'
    uint16_t version;
    uint16_t reserved;
    uint32_t steps;
    uint32_t batch_size;
    uint32_t action_count;
    uint32_t state_size;
    double   epsilon;
    double   temperature;
} TrajFileHeader;

int write_batch_file(const char* path_tmp, const char* path_final, trajectory** batch, int batch_size, int steps, double epsilon, double temperature);
int read_batch_file(const char* path, trajectory*** out_batch, int* out_batch_size, int* out_steps, double* out_epsilon, double* out_temperature);

#endif