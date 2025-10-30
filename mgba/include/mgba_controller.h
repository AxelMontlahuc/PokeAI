#ifndef MGBA_CONTROLLER_H
#define MGBA_CONTROLLER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mgba_connection.h"

#ifndef _WIN32
    #include <sys/select.h>
    #include <sys/time.h>
    static inline void Sleep(unsigned int ms) {
        struct timeval tv;
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        (void)select(0, NULL, NULL, NULL, &tv);
    }
#endif

typedef enum {
    MGBA_BUTTON_UP,
    MGBA_BUTTON_DOWN,
    MGBA_BUTTON_LEFT,
    MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_A,
    MGBA_BUTTON_B,
    MGBA_BUTTON_START
} MGBAButton;

int mgba_reset(MGBAConnection* conn);
int mgba_press_button(MGBAConnection* conn, MGBAButton button, int delay_ms);

#endif