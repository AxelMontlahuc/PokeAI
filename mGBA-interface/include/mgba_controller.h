#ifndef MGBA_CONTROLLER_H
#define MGBA_CONTROLLER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mgba_connection.h"
#include <windows.h>

typedef enum {
    MGBA_BUTTON_A,
    MGBA_BUTTON_B,
    MGBA_BUTTON_SELECT,
    MGBA_BUTTON_START,
    MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_LEFT,
    MGBA_BUTTON_UP,
    MGBA_BUTTON_DOWN,
    MGBA_BUTTON_R,
    MGBA_BUTTON_L
} MGBAButton;
int mgba_press_button(MGBAConnection* conn, MGBAButton button, int delay_ms);
int mgba_hold_button(MGBAConnection* conn, MGBAButton button, int duration_frames);

#endif