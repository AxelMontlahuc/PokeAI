/**
 * test_state_mgba.c - Test state reading from mGBA via socket
 * 
 * This standalone program connects to mGBA running socket.lua and fetches
 * all state information to verify memory addresses are correct.
 * 
 * Usage: ./bin/test_state_mgba [port]
 *        Default port: 8888
 * 
 * Requirements:
 *   1. mGBA running with socket.lua script loaded
 *   2. A game loaded in mGBA
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mgba/include/mgba_connection.h"
#include "../mgba/include/mgba_intel.h"
#include "constants.h"

// Print a 11x11 behavior map with visual representation
static void print_behavior_map(int behavior[11][11], int player_x, int player_y) {
    printf("\n=== BEHAVIOR MAP (11x11 centered on player) ===\n");
    printf("Player position: (%d, %d)\n", player_x, player_y);
    printf("Legend: # = solid(-1), . = walkable(0), ~ = grass(1), ! = interactable(2), @ = player\n\n");
    
    for (int r = 0; r < 11; r++) {
        printf("  ");
        for (int c = 0; c < 11; c++) {
            if (r == 5 && c == 5) {
                printf("@ ");  // Player is always at center
            } else {
                switch (behavior[r][c]) {
                    case -1: printf("# "); break;  // Solid
                    case  0: printf(". "); break;  // Walkable
                    case  1: printf("~ "); break;  // Grass
                    case  2: printf("! "); break;  // Interactable
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

int main(int argc, char* argv[]) {
    int port = 8888;
    const char* host = "127.0.0.1";
    
    if (argc > 1) {
        port = atoi(argv[1]);
    }
    
    printf("===========================================\n");
    printf("   PokeAI State Test (mGBA Socket)\n");
    printf("===========================================\n");
    printf("Connecting to %s:%d\n", host, port);
    
    // Connect to mGBA
    MGBAConnection conn;
    printf("\nConnecting to mGBA...\n");
    if (mgba_connect(&conn, host, port) != 0) {
        fprintf(stderr, "Failed to connect to mGBA at %s:%d\n", host, port);
        fprintf(stderr, "Make sure mGBA is running with socket.lua loaded!\n");
        return 1;
    }
    printf("Connected!\n");
    
    // Read behavior map
    printf("\n--- Reading behavior map ---\n");
    int behavior[11][11];
    int player_x, player_y;
    
    if (read_behavior_map(conn.sock, behavior, &player_x, &player_y) != 0) {
        fprintf(stderr, "Failed to read behavior map\n");
    } else {
        print_behavior_map(behavior, player_x, player_y);
    }
    
    // Read state
    printf("\n--- Reading state via read_state() ---\n");
    int team[6 * 8];
    int enemy[3];
    int pp[4];
    int zone, clock;
    int bg0[32][32];
    int bg2[32][32];
    
    if (read_state(conn.sock, team, enemy, pp, &zone, &clock, bg0, bg2) != 0) {
        fprintf(stderr, "Failed to read state\n");
    } else {
        print_team(team);
        print_enemy(enemy);
        print_pp(pp);
        print_misc(zone, clock);
    }
    
    // Disconnect
    mgba_disconnect(&conn);
    
    printf("\n===========================================\n");
    printf("   Test complete!\n");
    printf("===========================================\n");
    
    return 0;
}
