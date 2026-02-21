/*
 * skipintro - Spam A button through the Pokémon Emerald intro,
 * then save a savestate.
 *
 * Usage: ./bin/skipintro [num_presses] [output_path]
 *        Default: 500 presses, saves to saves/clock.sav
 */

#include <stdio.h>
#include <stdlib.h>
#include "../gba/gba.h"
#include "constants.h"

int main(int argc, char** argv) {
    int num_presses = 100;
    const char* output = SAVESTATE_PATH;

    if (argc >= 2) num_presses = atoi(argv[1]);
    if (argc >= 3) output = argv[2];

    printf("=== Skip Intro ===\n");
    printf("ROM: %s\n", ROM_PATH);
    printf("Core: %s\n", CORE_PATH);
    printf("Presses: %d (A button, %d frames apart)\n", num_presses, SPEED);
    printf("Output: %s\n\n", output);

    if (gba_create(CORE_PATH, ROM_PATH) != 0) {
        fprintf(stderr, "Failed to create GBA emulator\n");
        return 1;
    }

    // Run a few frames to let the emulator initialize
    printf("Warming up...\n");
    gba_run(60);

    printf("Spamming A button...\n");
    for (int i = 0; i < num_presses; i++) {
        gba_button(4);  // 4 = A button
        gba_run(SPEED);

        if ((i + 1) % 50 == 0) {
            printf("  Press %d/%d\n", i + 1, num_presses);
            // Save a screenshot every 50 presses to monitor progress
            char path[256];
            snprintf(path, sizeof(path), "screen/skipintro_%d.bmp", i + 1);
            gba_screen(path);
        }
    }

    printf("\nSaving savestate to %s ...\n", output);
    gba_savestate(output);
    gba_screen("screen/skipintro_final.bmp");

    printf("Done! Check screen/skipintro_final.bmp to see where you ended up.\n");
    printf("If you need more presses, run again with a higher count.\n");

    gba_destroy();
    return 0;
}
