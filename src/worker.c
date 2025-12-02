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
#include <sys/time.h>
#endif

#include "state.h"
#include "reward.h"
#include "policy.h"
#include "constants.h"
#include "checkpoint.h"
#include "serializer.h"
#include "../gba/gba.h"

static void ensure_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static int read_phase_file(const char* locks_dir) {
    char path[512];
#ifdef _WIN32
    snprintf(path, sizeof(path), "%s\\phase", locks_dir);
#else
    snprintf(path, sizeof(path), "%s/phase", locks_dir);
#endif
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int p = 0;
    if (fscanf(f, "%d", &p) != 1) { fclose(f); return 0; }
    fclose(f);
    return p;
}

static void write_ready_flag(const char* locks_dir, int phase, int id) {
    char tmp[512], final[512];
#ifdef _WIN32
    snprintf(tmp, sizeof(tmp),   "%s\\ready-%06d-%03d.tmp",  locks_dir, phase, id);
    snprintf(final, sizeof(final),"%s\\ready-%06d-%03d.flag", locks_dir, phase, id);
#else
    snprintf(tmp, sizeof(tmp),   "%s/ready-%06d-%03d.tmp",  locks_dir, phase, id);
    snprintf(final, sizeof(final),"%s/ready-%06d-%03d.flag", locks_dir, phase, id);
#endif
    FILE* f = fopen(tmp, "wb");
    if (f) { fclose(f); rename(tmp, final); }
}

static void wait_for_phase_advance(const char* locks_dir, int cur_phase) {
    for (;;) {
#ifdef _WIN32
        Sleep(50);
#else
        usleep(50 * 1000);
#endif
        int p = read_phase_file(locks_dir);
        if (p > cur_phase) break;
    }
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

        gba_button(traj->actions[i]);
        gba_run(SPEED);
        gba_screen(SCREEN_PATH);

        state s_next = fetchState();
        traj->rewards[i] = pnl(traj->states[i], s_next);

    }

    free(input_vec);

    return traj;
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        ID = atoi(argv[1]);
    }
    snprintf(SCREEN_PATH, sizeof(SCREEN_PATH), "%s%d.bmp", SCREEN_PATH_PREFIX, ID);

    ensure_dir(CHECKPOINT_DIR);
    ensure_dir(QUEUE_DIR);
    ensure_dir(LOCKS_DIR);
    ensure_dir(SAVES_DIR);
    ensure_dir(SCREEN_DIR);
    ensure_dir(LOGS_DIR);

    unsigned int seed = (unsigned int)(time(NULL) * ID);
    srand(seed);

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(CHECKPOINT_PATH, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(INPUT_SIZE, HIDDEN_SIZE, ACTION_COUNT);

    int episode = (int)loaded_episodes;
    double temperature = TEMP_MIN;

    int batches_per_episode = (WORKER_TRAJECTORIES + WORKER_BATCH_SIZE - 1) / WORKER_BATCH_SIZE; // On rajoute WORKER_BATCH_SIZE pour arrondir au supérieur
    int batches_done_in_episode = 0;

    (void)gba_create(CORE_PATH, ROM_PATH);

    while (1) {
        int phase = read_phase_file(LOCKS_DIR);

        uint64_t ep_tmp = 0ULL;
        uint64_t seed_tmp = 0ULL;
        LSTM* reloaded = loadLSTM(CHECKPOINT_PATH, &ep_tmp, &seed_tmp);
        if (reloaded) {
            if (network) freeLSTM(network);
            network = reloaded;
            episode = (int)ep_tmp;
            if (seed_tmp != 0ULL) {
                seed = (unsigned int)(time(NULL) * ID);
                srand(seed);
            }
            printf("[Worker] Reloaded checkpoint (episode=%d)\n", episode);
        }

        if (batches_done_in_episode == 0 || stopCondition()) {
            batches_done_in_episode = 0; 
            gba_reset(SAVESTATE_PATH);
            resetFlags();
        }

        temperature = fmax(TEMP_MIN, TEMP_MAX * pow(TEMP_DECAY, (double)episode));

        trajectory** batch = malloc(WORKER_BATCH_SIZE * sizeof(trajectory*));
        assert(batch != NULL);
        for (int b=0; b<WORKER_BATCH_SIZE; b++) {
            batch[b] = runTrajectory(network, WORKER_STEPS, temperature);
        }

        char tmp_path[512], final_path[512];
#ifdef _WIN32
        snprintf(tmp_path, sizeof(tmp_path), "%s\\phase-%06d-worker-%03d.traj.tmp", QUEUE_DIR, phase, ID);
        snprintf(final_path, sizeof(final_path), "%s\\phase-%06d-worker-%03d.traj", QUEUE_DIR, phase, ID);
#else
        snprintf(tmp_path, sizeof(tmp_path), "%s/phase-%06d-worker-%03d.traj.tmp", QUEUE_DIR, phase, ID);
        snprintf(final_path, sizeof(final_path), "%s/phase-%06d-worker-%03d.traj", QUEUE_DIR, phase, ID);
#endif

        if (write_batch_file(tmp_path, final_path, batch, WORKER_BATCH_SIZE, WORKER_STEPS, temperature) != 0) {
            printf("[Worker] Failed to write %s\n", final_path);
        } else {
            printf("[Worker] Enqueued %s\n", final_path);
        }
        for (int b=0; b<WORKER_BATCH_SIZE; b++) freeTrajectory(batch[b]);
        free(batch);

        write_ready_flag(LOCKS_DIR, phase, ID);
        wait_for_phase_advance(LOCKS_DIR, phase);

        batches_done_in_episode++;
        if (batches_done_in_episode >= batches_per_episode) {
            batches_done_in_episode = 0; 
        }
    }

    freeLSTM(network);
    gba_destroy();
    return 0;
}