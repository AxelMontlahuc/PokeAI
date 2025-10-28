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
#include "reward.h"
#include "policy.h"
#include "constants.h"
#include "checkpoint.h"
#include "serializer.h"
#include "../gba/gba.h"

const char* ROM_PATH = "/home/axel/Documents/Dev/PokeAI/ROM/pokemon.gba";
const char* CORE_PATH = "/home/axel/Documents/Dev/libretro-super/dist/unix/mgba_libretro.so";
const char* SCREEN_PATH = "/home/axel/Documents/Dev/PokeAI/screen/1.bmp";
const char* SAVESTATE_PATH = "/home/axel/Documents/Dev/PokeAI/ROM/start.sav";

const int SPEED = 120;

static void ensure_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static const int ACTIONS[ACTION_COUNT] = {
    0, 1, 2, 3, 4, 5 // UP, DOWN, LEFT, RIGHT, A, B
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

void startProcedure() {
    gba_run(300);
    for (int i=0; i<100; i++) {
        gba_button(4);
        gba_run(SPEED);
    }
    gba_screen(SCREEN_PATH);
}

trajectory* runTrajectory(LSTM* network, int steps, double temperature) {
    trajectory* traj = initTrajectory(steps);
    assert(traj != NULL);

    for (int j=0; j<network->hiddenSize; j++) {
        network->hiddenState[j] = 0.0;
        network->cellState[j] = 0.0;
    }

    double* input_vec = malloc(INPUT_SIZE * sizeof(double));
    assert(input_vec != NULL);

    for (int i=0; i<steps; i++) {
        traj->states[i] = fetchState();
        convertState(traj->states[i], input_vec);
        double* probs = forward(network, input_vec, temperature);

        traj->probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->probs[i] != NULL);
        for (int k=0; k<ACTION_COUNT; k++) traj->probs[i][k] = probs[k];
        

        traj->actions[i] =  ACTIONS[chooseAction(traj->probs[i], ACTION_COUNT)];
        traj->values[i] = network->last_value;

        gba_run(SPEED);

        gba_button(traj->actions[i]);
        gba_screen(SCREEN_PATH);

        state s_next = fetchState();
        traj->rewards[i] = pnl(traj->states[i], s_next);

    }

    free(input_vec);

    return traj;
}

int main(int argc, char** argv) {
    if (argc >= 2) PORT = atoi(argv[1]);

    ensure_dir("checkpoints");
    ensure_dir(QUEUE_DIR);

    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(CHECKPOINT_PATH, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(INPUT_SIZE, HIDDEN_SIZE, ACTION_COUNT);

    int episode = (int)loaded_episodes;
    double temperature = TEMP_MIN;

    int file_seq = 0;

    (void)gba_create(CORE_PATH, ROM_PATH);

    while (1) {
        uint64_t ep_tmp = 0ULL;
        uint64_t seed_tmp = 0ULL;
        LSTM* reloaded = loadLSTM(CHECKPOINT_PATH, &ep_tmp, &seed_tmp);
        if (reloaded) {
            if (network) freeLSTM(network);
            network = reloaded;
            episode = (int)ep_tmp;
            if (seed_tmp != 0ULL) { 
                seed = (unsigned int)seed_tmp; 
                srand(seed); 
            }
            printf("[Worker] Reloaded checkpoint (episode=%d)\n", episode);
        }

        gba_reset();
        startProcedure();
        resetFlags();

        temperature = fmax(TEMP_MIN, TEMP_MAX * pow(TEMP_DECAY, (double)episode));

        for (int t=0; t<WORKER_TRAJECTORIES; t += WORKER_BATCH_SIZE) {
            trajectory** batch = (trajectory**)malloc(WORKER_BATCH_SIZE * sizeof(trajectory*));
            assert(batch != NULL);

            for (int b=0; b<WORKER_BATCH_SIZE; b++) {
                batch[b] = runTrajectory(network, WORKER_STEPS, temperature);
            }

            char tmp_path[512], final_path[512];
#ifdef _WIN32
            snprintf(tmp_path, sizeof(tmp_path), "%s\\worker-%d-%06d.traj.tmp", QUEUE_DIR, PORT, file_seq);
            snprintf(final_path, sizeof(final_path), "%s\\worker-%d-%06d.traj", QUEUE_DIR, PORT, file_seq);
#else
            snprintf(tmp_path, sizeof(tmp_path), "%s/worker-%d-%06d.traj.tmp", QUEUE_DIR, PORT, file_seq);
            snprintf(final_path, sizeof(final_path), "%s/worker-%d-%06d.traj", QUEUE_DIR, PORT, file_seq);
#endif

            if (write_batch_file(tmp_path, final_path, batch, WORKER_BATCH_SIZE, WORKER_STEPS, temperature) != 0) {
                fprintf(stderr, "[Worker] Failed to write %s\n", final_path);
            } else {
                printf("[Worker] Enqueued %s\n", final_path);
            }
            file_seq++;

            for (int b=0; b<WORKER_BATCH_SIZE; b++) freeTrajectory(batch[b]);
            free(batch);
        }

        episode++;
    }

    freeLSTM(network);
    gba_destroy();
    return 0;
}