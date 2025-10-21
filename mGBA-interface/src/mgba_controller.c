#include "../include/mgba_controller.h"

const char* BUTTON_NAMES[] = {
    "Up", "Down", "Left", "Right", "A", "B", "Start"
};

const char* mgba_button_to_string(MGBAButton button) {
    if (button >= MGBA_BUTTON_UP && button <= MGBA_BUTTON_START) {
        return BUTTON_NAMES[button];
    }
    return "Unknown";
}

int mgba_reset(MGBAConnection* conn) {
    char message[] = "mgba-http.emu.reset";
    char response[1024];

    int result = mgba_send_command(conn, message, response, sizeof(response));
    if (result < 0) {
        return result;
    }

    return 0;
}

int mgba_press_button(MGBAConnection* conn, MGBAButton button, int delay_ms) {
    char message[64];
    char response[1024];
    const char* button_str = mgba_button_to_string(button);

    snprintf(message, sizeof(message), "mgba-http.button.tap,%s", button_str);

    int result = mgba_send_command(conn, message, response, sizeof(response));
    if (result < 0) {
        return result;
    }

    if (delay_ms > 0) {
        Sleep(delay_ms);
    }
    
    return 0;
}

int mgba_hold_button(MGBAConnection* conn, MGBAButton button, int duration_frames) {
    char message[64];
    char response[1024];
    const char* button_str = mgba_button_to_string(button);

    snprintf(message, sizeof(message), "mgba-http.button.hold,%s,%d", button_str, duration_frames);

    int result = mgba_send_command(conn, message, response, sizeof(response));
    if (result < 0) {
        return result;
    }
    
    return 0;
}