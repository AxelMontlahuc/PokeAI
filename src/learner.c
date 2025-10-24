#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif

#include "struct.h"
#include "func.h"
#include "state.h"
#include "reward.h"
#include "policy.h"
#include "checkpoint.h"
#include "serializer.h"

#ifdef _WIN32
#include <windows.h>
#endif

static void ensure_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static int list_traj_files(const char* dir, char files[][512], int maxn) {
    int n = 0;
#ifdef _WIN32
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s\\*.traj", dir);
    struct _finddata_t f; intptr_t h = _findfirst(pattern, &f);
    if (h == -1) return 0;
    do {
        if (n >= maxn) break;
        snprintf(files[n], 512, "%s\\%s", dir, f.name);
        n++;
    } while (_findnext(h, &f) == 0);
    _findclose(h);
#else
    DIR* d = opendir(dir);
    if (!d) return 0;
    struct dirent* e;
    while ((e = readdir(d)) && n < maxn) {
        if (strstr(e->d_name, ".traj")) {
            snprintf(files[n], 512, "%s/%s", dir, e->d_name);
            n++;
        }
    }
    closedir(d);
#endif
    return n;
}

int main() {
    const char* queue_dir = "queue";
    const char* checkpoint_path = "checkpoints/model-last.sav";
    int files_per_step = 4;

    ensure_dir("checkpoints");
    ensure_dir(queue_dir);

    int inputSize = 6*8 + 4 + 3 + 2 + 2*32*32;
    int hiddenSize = 128;

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(checkpoint_path, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(inputSize, hiddenSize, ACTION_COUNT);

    unsigned int seed = (unsigned int)((loaded_seed != 0) ? (unsigned int)loaded_seed : (unsigned int)time(NULL));
    srand(seed);

    int episode = (int)loaded_episodes;

    while (1) {
        char files[64][512];
        int n = list_traj_files(queue_dir, files, files_per_step);
        if (n == 0) {
#ifdef _WIN32
            Sleep(50);
#else
            usleep(50 * 1000);
#endif
            continue;
        }

        trajectory*** batches = malloc(sizeof(trajectory**) * n);

        int* b_sizes = malloc(sizeof(int) * n);
        int* b_steps = malloc(sizeof(int) * n);
        double* b_eps = malloc(sizeof(double) * n);
        double* b_temp = malloc(sizeof(double) * n);

        int total_traj = 0;
        int steps = -1;

        for (int i = 0; i < n; i++) {
            trajectory** batch = NULL;

            int batch_size = 0;
            int s = 0;
            double eps = 0.0;
            double temp = 1.0;

            if (read_batch_file(files[i], &batch, &batch_size, &s, &eps, &temp) != 0) {
                fprintf(stderr, "[Learner] Failed to read %s\n", files[i]);
                continue;
            }

            batches[i] = batch;
            b_sizes[i] = batch_size;
            b_steps[i] = s;
            b_eps[i] = eps;
            b_temp[i] = temp;
            total_traj += batch_size;

            if (steps < 0) steps = s;
        }

        if (total_traj == 0) {
            goto cleanup;
        }

        trajectory** flat = malloc(sizeof(trajectory*) * total_traj);
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < b_sizes[i]; j++) {
                flat[idx++] = batches[i][j];
            }
        }

        double epsilon = 0.2;
        double temperature = 1.0;
        for (int i = 0; i < n; i++) {
            if (b_sizes[i] > 0) { 
                epsilon = b_eps[i]; 
                temperature = b_temp[i]; 
                break; 
            }
        }

        printf("[Learner] Training on %d traj (steps=%d) from %d files\n", total_traj, steps, n);

        backpropagation(network, 0.01, steps, flat, total_traj, temperature, epsilon);

        episode++;
        if (saveLSTMCheckpoint(checkpoint_path, network, (uint64_t)episode, (uint64_t)seed) == 0) {
            printf("[Learner] Saved checkpoint (episode=%d)\n", episode);
        }

        for (int i = 0; i < total_traj; i++) freeTrajectory(flat[i]);
        free(flat);
        for (int i = 0; i < n; i++) {
            if (batches[i]) free(batches[i]);
            remove(files[i]);
        }

    cleanup:
        free(batches); 
        free(b_sizes); 
        free(b_steps); 
        free(b_eps); 
        free(b_temp);
    }

    freeLSTM(network);
    return 0;
}