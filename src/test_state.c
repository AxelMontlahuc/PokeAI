/**
 * test_state.c - Test state reading from libretro GBA emulator
 * 
 * This standalone program loads a ROM and savestate, then fetches and prints
 * all state information to verify memory addresses are correct.
 * 
 * Usage: ./bin/test_state [savestate_path]
 *        Default savestate: ROM/start.ss0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../gba/gba.h"
#include "constants.h"

// Print a 11x11 behavior map with visual representation
static void print_behavior_map(int behavior[11][11], int player_x, int player_y) {
    printf("\n=== BEHAVIOR MAP (11x11 centered on player) ===\n");
    printf("Player position: (%d, %d)\n", player_x, player_y);
    printf("Legend: # = solid(-1), . = walkable(0), ! = interactable(1), @ = player\n\n");
    
    for (int r = 0; r < 11; r++) {
        printf("  ");
        for (int c = 0; c < 11; c++) {
            if (r == 5 && c == 5) {
                printf("@ ");  // Player is always at center
            } else {
                switch (behavior[r][c]) {
                    case -1: printf("# "); break;  // Solid
                    case  0: printf(". "); break;  // Walkable
                    case  1: printf("! "); break;  // Interactable
                    default: printf("? "); break;  // Unknown
                }
            }
        }
        printf("\n");
    }
    
    printf("\nRaw values:\n");
    for (int r = 0; r < 11; r++) {
        printf("  ");
        for (int c = 0; c < 11; c++) {
            printf("%2d ", behavior[r][c]);
        }
        printf("\n");
    }
}

// Print team Pokémon stats
static void print_team(int* team) {
    printf("\n=== TEAM POKEMON ===\n");
    for (int i = 0; i < 6; i++) {
        int o = i * 8;
        int maxHP = team[o + 0];
        int HP = team[o + 1];
        int level = team[o + 2];
        int ATK = team[o + 3];
        int DEF = team[o + 4];
        int SPEED = team[o + 5];
        int ATK_SPE = team[o + 6];
        int DEF_SPE = team[o + 7];
        
        if (level == 0 && maxHP == 0) {
            printf("  Slot %d: (empty)\n", i + 1);
        } else {
            printf("  Slot %d: Lv.%d  HP: %d/%d  ATK:%d DEF:%d SPD:%d SPATK:%d SPDEF:%d\n",
                   i + 1, level, HP, maxHP, ATK, DEF, SPEED, ATK_SPE, DEF_SPE);
        }
    }
}

// Print enemy stats
static void print_enemy(int* enemy) {
    printf("\n=== ENEMY POKEMON ===\n");
    if (enemy[2] == 0 && enemy[0] == 0) {
        printf("  No enemy in battle\n");
    } else {
        printf("  Lv.%d  HP: %d/%d\n", enemy[2], enemy[1], enemy[0]);
    }
}

// Print move PP
static void print_pp(int* pp) {
    printf("\n=== MOVE PP ===\n");
    printf("  Move 1: %d  Move 2: %d  Move 3: %d  Move 4: %d\n",
           pp[0], pp[1], pp[2], pp[3]);
}

// Print zone and clock
static void print_misc(int zone, int clock) {
    printf("\n=== MISC DATA ===\n");
    printf("  Zone ID: %d (0x%04X)\n", zone, zone);
    printf("  Clock: %d\n", clock);
}

// Debug: print raw memory values at key addresses
static void print_debug_addresses(void) {
    printf("\n=== DEBUG: RAW MEMORY ADDRESSES ===\n");
    
    // Map layout pointer
    long layout_ptr = gba_ram(0x02037318, 4);
    printf("  Map Layout Ptr (0x02037318): 0x%08lX\n", layout_ptr);
    
    if (layout_ptr > 0) {
        // Read layout structure
        long map_width = gba_ram((uint32_t)layout_ptr + 0x00, 4);
        long map_height = gba_ram((uint32_t)layout_ptr + 0x04, 4);
        long map_data = gba_ram((uint32_t)layout_ptr + 0x08, 4);
        long primary_ts = gba_ram((uint32_t)layout_ptr + 0x0C, 4);
        long secondary_ts = gba_ram((uint32_t)layout_ptr + 0x10, 4);
        
        printf("    -> Map Width: %ld\n", map_width);
        printf("    -> Map Height: %ld\n", map_height);
        printf("    -> Map Data Ptr: 0x%08lX\n", map_data);
        printf("    -> Primary Tileset: 0x%08lX\n", primary_ts);
        printf("    -> Secondary Tileset: 0x%08lX\n", secondary_ts);
    }
    
    // Player position
    long player_x = gba_ram(0x020322E0, 2);
    long player_y = gba_ram(0x020322E2, 2);
    printf("  Player X (0x020322E0): %ld\n", player_x);
    printf("  Player Y (0x020322E2): %ld\n", player_y);
    
    // Zone
    long zone = gba_ram(0x020322E4, 2);
    printf("  Zone (0x020322E4): %ld (0x%04lX)\n", zone, zone);
}

int main(int argc, char* argv[]) {
    const char* savestate = "ROM/start.ss0";
    
    if (argc > 1) {
        savestate = argv[1];
    }
    
    printf("===========================================\n");
    printf("   PokeAI State Test (libretro GBA)\n");
    printf("===========================================\n");
    printf("ROM: %s\n", ROM_PATH);
    printf("Core: %s\n", CORE_PATH);
    printf("Savestate: %s\n", savestate);
    
    // Initialize emulator
    printf("\nInitializing emulator...\n");
    if (gba_create(CORE_PATH, ROM_PATH) != 0) {
        fprintf(stderr, "Failed to create GBA emulator\n");
        return 1;
    }
    
    // Load savestate
    printf("Loading savestate...\n");
    if (gba_reset(savestate) != 0) {
        fprintf(stderr, "Failed to load savestate: %s\n", savestate);
        gba_destroy();
        return 1;
    }
    
    // Run a few frames to stabilize
    printf("Running 10 frames to stabilize...\n");
    gba_run(10);
    
    // Print debug addresses first
    print_debug_addresses();
    
    // Fetch state
    printf("\n--- Fetching state via gba_state() ---\n");
    
    int team[6 * 8];
    int enemy[3];
    int pp[4];
    int zone, clock;
    int behavior[11][11];
    int player_x, player_y;
    
    gba_state(team, enemy, pp, &zone, &clock, behavior, &player_x, &player_y);
    
    // Print all state data
    print_team(team);
    print_enemy(enemy);
    print_pp(pp);
    print_misc(zone, clock);
    print_behavior_map(behavior, player_x, player_y);
    
    // Cleanup
    gba_destroy();
    
    printf("\n===========================================\n");
    printf("   Test complete!\n");
    printf("===========================================\n");
    
    return 0;
}
