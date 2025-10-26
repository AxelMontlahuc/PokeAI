#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "state.h"
#include "constants.h"

int write_batch_file(const char* path_tmp, const char* path_final, trajectory** batch, int batch_size, int steps, double temperature);
int read_batch_file(const char* path, trajectory*** out_batch, int* out_batch_size, int* out_steps, double* out_temperature);

#endif