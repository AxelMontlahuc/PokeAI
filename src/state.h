#ifndef STATE_H
#define STATE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "struct.h"

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_intel.h"
#include "../mGBA-interface/include/mgba_map.h"

pokemon fetchPoke(MGBAConnection conn, int i);
state fetchState(MGBAConnection conn);
double* convertState(state s);

#endif