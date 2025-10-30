#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "state.h"
#include "policy.h"
#include "constants.h"
#include "checkpoint.h"

#include "../mgba/include/mgba_connection.h"
#include "../mgba/include/mgba_controller.h"

static const MGBAButton ACTIONS[ACTION_COUNT] = {
    MGBA_BUTTON_UP, MGBA_BUTTON_DOWN, MGBA_BUTTON_LEFT, MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_A, MGBA_BUTTON_B
};

static int chooseAction(double* p, int n) {
    double r = (double)rand() / RAND_MAX;
    double c = 0.0;
    for (int i = 0; i < n; i++) {
        c += p[i];
        if (r < c || i == n - 1) return i;
    }
    return n - 1;
}

int main(int argc, char** argv) {
    if (argc >= 2) PORT = atoi(argv[1]);
    if (argc >= 3) CHECKPOINT_PATH = argv[2];

    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(CHECKPOINT_PATH, &loaded_episodes, &loaded_seed);
    if (!network) {
        printf("[Worker] Failed to load checkpoint %s\n", CHECKPOINT_PATH);
        return 1;
    }

    MGBAConnection conn;
    if (mgba_connect(&conn, HOST_ADDR, PORT) == 0) {
        printf("[Worker] Connected to mGBA %s:%d\n", HOST_ADDR, PORT);
    } else {
        printf("[Worker] Failed to connect mGBA %s:%d\n", HOST_ADDR, PORT);
        return 1;
    }

    mgba_reset(&conn);

    while (1) {
        double* s_vec = malloc(INPUT_SIZE * sizeof(double));
        assert(s_vec != NULL);

        state s = fetchMGBAState(conn);
        convertState(s, s_vec);

        double* probs = forward(network, s_vec, 1.0);
        int action_idx = chooseAction(probs, ACTION_COUNT);
        MGBAButton action = ACTIONS[action_idx];

        mgba_press_button(&conn, action, BUTTON_PRESS_MS);

        free(s_vec);
        free(probs);
    }

    freeLSTM(network);
    mgba_disconnect(&conn);
    return 0;
}