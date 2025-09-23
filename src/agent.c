#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_controller.h"
#include "../mGBA-interface/include/mgba_map.h"
#include "../mGBA-interface/include/mgba_intel.h"


double* fetchState(MGBAConnection conn) {
    double* state = malloc((4*(32*32) + (6*8) + 4 + 3 + 1) * sizeof(double));
    MGBAMap* map;
    for (int i=0; i<4; i++) {
        map = mgba_read_map(&conn, i);
        for (int k=0; k<32; k++) {
            for (int l=0; l<32; l++) {
                state[i*32*32 + k*32 + l] = (double)map->tiles[k][l];
            }
        }
    }
    mgba_free_map(map);
    
    for (int i=0; i<6; i++) {
        state[4*(32*32) + i*8 + 0] = (double)get_max_HP(conn.sock, i);
        state[4*(32*32) + i*8 + 1] = (double)get_HP(conn.sock, i);
        state[4*(32*32) + i*8 + 2] = (double)get_level(conn.sock, i);
        state[4*(32*32) + i*8 + 3] = (double)get_ATK(conn.sock, i);
        state[4*(32*32) + i*8 + 4] = (double)get_DEF(conn.sock, i);
        state[4*(32*32) + i*8 + 5] = (double)get_SPEED(conn.sock, i);
        state[4*(32*32) + i*8 + 6] = (double)get_ATK_SPE(conn.sock, i);
        state[4*(32*32) + i*8 + 7] = (double)get_DEF_SPE(conn.sock, i);
    }

    state[4*(32*32) + 6*8 + 0] = (double)get_PP(conn.sock, 0);
    state[4*(32*32) + 6*8 + 1] = (double)get_PP(conn.sock, 1);
    state[4*(32*32) + 6*8 + 2] = (double)get_PP(conn.sock, 2);
    state[4*(32*32) + 6*8 + 3] = (double)get_PP(conn.sock, 3);

    state[4*(32*32) + 6*8 + 4] = (double)get_enemy_max_HP(conn.sock);
    state[4*(32*32) + 6*8 + 5] = (double)get_enemy_HP(conn.sock);
    state[4*(32*32) + 6*8 + 6] = (double)get_enemy_level(conn.sock);

    state[4*(32*32) + 6*8 + 7] = (double)get_zone(conn.sock);

    return state;
}

double softmax(double* logits, int n, int i) {
    double sum = 0.0;
    for (int j=0; j<n; j++) {
        sum += exp(logits[j]);
    }
    return exp(logits[i]) / sum;
}

int main() {
    MGBAConnection conn;
    int result;
    result = mgba_connect(&conn, "127.0.0.1", 8888);
    if (result != 0) {
        printf("Failed to connect to mGBA. Error code: %d\n", result);
        return 1;
    }
    printf("Connected to mGBA.\n");

    return 0;
}