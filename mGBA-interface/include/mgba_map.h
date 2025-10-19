#ifndef MGBA_MAP_H
#define MGBA_MAP_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mgba_connection.h"
#include <windows.h>

typedef struct {
    int tile_id;
    char symbol;
} MGBATileMapping;
typedef struct {
    int width;
    int height;
    int** tiles;
} MGBAMap;

MGBAMap* mgba_read_bg0(MGBAConnection* conn);
MGBAMap* mgba_read_map(MGBAConnection* conn, int bg);
void mgba_free_map(MGBAMap* map);

#endif