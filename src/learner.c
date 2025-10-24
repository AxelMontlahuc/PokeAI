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

static int cmp_double_asc(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main() {
    const char* queue_dir = "queue";
    const char* checkpoint_path = "checkpoints/model-last.sav";
    const char* locks_dir = "locks";
    int files_per_step = 4;

    ensure_dir("checkpoints");
    ensure_dir(queue_dir);

#ifdef _WIN32
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "cmd /C rmdir /S /Q \"%s\" >NUL 2>&1 & mkdir \"%s\" >NUL 2>&1", queue_dir, queue_dir);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "cmd /C rmdir /S /Q \"%s\" >NUL 2>&1 & mkdir \"%s\" >NUL 2>&1", locks_dir, locks_dir);
    system(cmd);
#else
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "rm -rf \"%s\" \"%s\"; mkdir -p \"%s\" \"%s\"", queue_dir, locks_dir, queue_dir, locks_dir);
    system(cmd);
#endif

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

        double sum_step_rewards = 0.0;
        double sum_traj_returns = 0.0;
        double sum_entropy = 0.0;
        int count_steps = total_traj * steps;
        for (int i = 0; i < total_traj; i++) {
            double ret = 0.0;
            for (int t = 0; t < steps; t++) {
                ret += flat[i]->rewards[t];
                double H = 0.0;
                for (int k = 0; k < ACTION_COUNT; k++) {
                    double p = flat[i]->probs[t][k];
                    if (p > 0.0) H -= p * log(p);
                }
                sum_entropy += H;
            }
            sum_traj_returns += ret;
            sum_step_rewards += ret;
        }
        double avg_entropy = (count_steps > 0) ? (sum_entropy / (double)count_steps) : 0.0;
        double avg_traj_return = (total_traj > 0) ? (sum_traj_returns / (double)total_traj) : 0.0;
        double avg_step_reward = (count_steps > 0) ? (sum_step_rewards / (double)count_steps) : 0.0;

        int action_counts[ACTION_COUNT] = {0};
        for (int i = 0; i < total_traj; i++) {
            for (int t = 0; t < steps; t++) {
                int idx;
                switch (flat[i]->actions[t]) {
                    case MGBA_BUTTON_UP: idx = 0; break;
                    case MGBA_BUTTON_DOWN: idx = 1; break;
                    case MGBA_BUTTON_LEFT: idx = 2; break;
                    case MGBA_BUTTON_RIGHT: idx = 3; break;
                    case MGBA_BUTTON_A: idx = 4; break;
                    case MGBA_BUTTON_B: idx = 5; break;
                    default: idx = 5; break;
                }
                action_counts[idx]++;
            }
        }

        double* traj_returns = malloc(sizeof(double) * total_traj);
        for (int i = 0; i < total_traj; i++) {
            double rsum = 0.0;
            for (int t = 0; t < steps; t++) rsum += flat[i]->rewards[t];
            traj_returns[i] = rsum;
        }
        qsort(traj_returns, total_traj, sizeof(double), cmp_double_asc);
        double p10 = traj_returns[(int)(0.10 * (total_traj - 1))];
        double p50 = traj_returns[(int)(0.50 * (total_traj - 1))];
        double p90 = traj_returns[(int)(0.90 * (total_traj - 1))];

        int total_steps = total_traj * steps;
        printf("\n[Learner] Update\n");
        printf("  Batch     : traj=%-4d steps=%-4d files=%-3d  eps=%-5.3f  temp=%-5.3f\n", total_traj, steps, n, epsilon, temperature);
        printf("  Rewards   : avg/step=%-8.5f  avg/traj=%-8.5f  p10=%-8.5f  p50=%-8.5f  p90=%-8.5f\n", avg_step_reward, avg_traj_return, p10, p50, p90);
        printf("  Entropy   : H=%-7.4f (mean across steps)\n", avg_entropy);
        double upP = (total_steps>0)? (100.0 * (double)action_counts[0]/(double)total_steps) : 0.0;
        double dnP = (total_steps>0)? (100.0 * (double)action_counts[1]/(double)total_steps) : 0.0;
        double lfP = (total_steps>0)? (100.0 * (double)action_counts[2]/(double)total_steps) : 0.0;
        double rtP = (total_steps>0)? (100.0 * (double)action_counts[3]/(double)total_steps) : 0.0;
        double aP  = (total_steps>0)? (100.0 * (double)action_counts[4]/(double)total_steps) : 0.0;
        double bP  = (total_steps>0)? (100.0 * (double)action_counts[5]/(double)total_steps) : 0.0;
        printf("  Actions   : Up=%5.1f%%  Down=%5.1f%%  Left=%5.1f%%  Right=%5.1f%%  A=%5.1f%%  B=%5.1f%%\n",
            upP, dnP, lfP, rtP, aP, bP);

        BackpropStats st = {0};
        clock_t t0 = clock();
        backpropagation(network, 0.01, steps, flat, total_traj, temperature, epsilon, &st);
        clock_t t1 = clock();
        double secs = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
        double sps = (secs > 0.0) ? ((double)total_steps / secs) : 0.0;
        printf("  Gradients : ||g||_2=%-9.4f  clip=%-6.3f  time=%-6.3fs  steps/s=%-8.1f\n\n", st.grad_norm, st.clip_scale, secs, sps);

        free(traj_returns);

        episode++;
        saveLSTMCheckpoint(checkpoint_path, network, (uint64_t)episode, (uint64_t)seed);

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