#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
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
#include "constants.h"

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

static int actionToIndex(MGBAButton action) {
    switch (action) {
        case MGBA_BUTTON_UP: return 0;
        case MGBA_BUTTON_DOWN: return 1;
        case MGBA_BUTTON_LEFT: return 2;
        case MGBA_BUTTON_RIGHT: return 3;
        case MGBA_BUTTON_A: return 4;
        case MGBA_BUTTON_B: return 5;
        case MGBA_BUTTON_START: return 6;
        default: return 5;
    }
}

int main() {
    ensure_dir("checkpoints");
    ensure_dir(QUEUE_DIR);

#ifdef _WIN32
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "cmd /C rmdir /S /Q \"%s\" >NUL 2>&1 & mkdir \"%s\" >NUL 2>&1", QUEUE_DIR, QUEUE_DIR);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "cmd /C rmdir /S /Q \"%s\" >NUL 2>&1 & mkdir \"%s\" >NUL 2>&1", LOCKS_DIR, LOCKS_DIR);
    system(cmd);
#else
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "rm -rf \"%s\" \"%s\"; mkdir -p \"%s\" \"%s\"", QUEUE_DIR, LOCKS_DIR, QUEUE_DIR, LOCKS_DIR);
    system(cmd);
#endif

    int inputSize = INPUT_SIZE;
    int hiddenSize = HIDDEN_SIZE;

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(CHECKPOINT_PATH, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(inputSize, hiddenSize, ACTION_COUNT);

    unsigned int seed = (unsigned int)((loaded_seed != 0) ? (unsigned int)loaded_seed : (unsigned int)time(NULL));
    srand(seed);

    int episode = (int)loaded_episodes;

    while (1) {
        char files[64][512];
    int n = list_traj_files(QUEUE_DIR, files, FILES_PER_STEP);
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
        double* b_temp = malloc(sizeof(double) * n);

        int total_traj = 0;
        int steps = -1;

        for (int i = 0; i < n; i++) {
            trajectory** batch = NULL;

            int batch_size = 0;
            int s = 0;
            double temp = 1.0;

            if (read_batch_file(files[i], &batch, &batch_size, &s, &temp) != 0) {
                fprintf(stderr, "[Learner] Failed to read %s\n", files[i]);
                continue;
            }

            batches[i] = batch;
            b_sizes[i] = batch_size;
            b_steps[i] = s;
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

        double temperature = 1.0;
        for (int i = 0; i < n; i++) {
            if (b_sizes[i] > 0) { 
                temperature = b_temp[i]; 
                break; 
            }
        }

        double sum_step_rewards = 0.0;
        double sum_traj_returns = 0.0;
        double sum_entropy = 0.0;
        double sum_values = 0.0;
        double sumsq_values = 0.0;
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
                sum_values += flat[i]->values[t];
                sumsq_values += flat[i]->values[t] * flat[i]->values[t];
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
        printf("  Batch     : traj=%-4d steps=%-4d files=%-3d  temp=%-5.3f\n", total_traj, steps, n, temperature);
        printf("  Rewards   : avg/step=%-8.5f  avg/traj=%-8.5f  p10=%-8.5f  p50=%-8.5f  p90=%-8.5f\n", avg_step_reward, avg_traj_return, p10, p50, p90);
        printf("  Entropy   : H=%-7.4f (mean across steps)\n", avg_entropy);
        if (count_steps > 0) {
            double mean_v = sum_values / (double)count_steps;
            double var_v = fmax(0.0, (sumsq_values / (double)count_steps) - (mean_v * mean_v));
            printf("  Values    : mean=%-8.5f  std=%-8.5f\n", mean_v, sqrt(var_v));
        }
        
        double upP = (total_steps>0)? (100.0 * (double)action_counts[0]/(double)total_steps) : 0.0;
        double dnP = (total_steps>0)? (100.0 * (double)action_counts[1]/(double)total_steps) : 0.0;
        double lfP = (total_steps>0)? (100.0 * (double)action_counts[2]/(double)total_steps) : 0.0;
        double rtP = (total_steps>0)? (100.0 * (double)action_counts[3]/(double)total_steps) : 0.0;
        double aP  = (total_steps>0)? (100.0 * (double)action_counts[4]/(double)total_steps) : 0.0;
        double bP  = (total_steps>0)? (100.0 * (double)action_counts[5]/(double)total_steps) : 0.0;
        printf("  Actions   : Up=%5.1f%%  Down=%5.1f%%  Left=%5.1f%%  Right=%5.1f%%  A=%5.1f%%  B=%5.1f%%\n", upP, dnP, lfP, rtP, aP, bP);

        double* adv_flat = NULL;
        double** A_per_traj = NULL;
        double** R_per_traj = NULL;
        const double gamma = GAMMA_DISCOUNT;
        const double gae_lambda = GAE_LAMBDA;

        if (total_steps > 0) {
            A_per_traj = malloc(sizeof(double*) * total_traj);
            R_per_traj = malloc(sizeof(double*) * total_traj);
            adv_flat = malloc(sizeof(double) * total_steps);
            int cur = 0;
            for (int b = 0; b < total_traj; b++) {
                A_per_traj[b] = malloc(sizeof(double) * steps);
                R_per_traj[b] = malloc(sizeof(double) * steps);
                compute_gae(flat[b]->rewards, flat[b]->values, steps, gamma, gae_lambda, A_per_traj[b], R_per_traj[b]);
                for (int t = 0; t < steps; t++) {
                    adv_flat[cur++] = A_per_traj[b][t];
                }
            }
            normPNL(adv_flat, total_steps);
        }

        double warmup = (episode <= WARMUP_EPISODES) ? (fmax(MIN_WARMUP_FACTOR, (double)episode / (double)WARMUP_EPISODES)) : 1.0;
        double lr = BASE_LR * pow(LR_DECAY, (double)episode) * warmup;
        int mb_size = (total_traj >= MB_TRAJ_THRESHOLD) ? MB_SIZE_DEFAULT : total_traj;

        int* indices = (int*)malloc(sizeof(int) * total_traj);
        for (int i = 0; i < total_traj; i++) indices[i] = i;

        BackpropStats st = {0};
        clock_t t0 = clock();

        for (int e = 0; e < PPO_EPOCHS; e++) {
            for (int i = total_traj - 1; i > 0; --i) {
                int j = rand() % (i + 1);
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }
            for (int s = 0; s < total_traj; s += mb_size) {
                int count = (s + mb_size <= total_traj) ? mb_size : (total_traj - s);
                trajectory** mb = (trajectory**)malloc(sizeof(trajectory*) * count);
                for (int m = 0; m < count; m++) mb[m] = flat[indices[s + m]];
                backpropagation(network, lr, steps, mb, count, temperature, &st);
                free(mb);
            }
        }
        clock_t t1 = clock();
        double secs = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
        double sps = (secs > 0.0) ? ((double)(total_steps * PPO_EPOCHS) / secs) : 0.0;
        printf("  Gradients : ||g||_2=%-9.4f  clip=%-6.3f  lr=%-7.5f  time=%-6.3fs  steps/s=%-8.1f\n", st.grad_norm, st.clip_scale, lr, secs, sps);
        free(indices);

        if (total_steps > 0) {
            const double clip_eps = CLIP_EPS;
            double kl_sum = 0.0;
            double ratio_sum = 0.0;
            double ratio_sq_sum = 0.0;
            double ratio_min = DBL_MAX;
            double ratio_max = -DBL_MAX;
            long clip_count = 0;
            double surr_sum = 0.0;

            int flat_idx = 0;
            for (int i = 0; i < total_traj; i++) {
                for (int j = 0; j < network->hiddenSize; j++) {
                    network->hiddenState[j] = 0.0;
                    network->cellState[j] = 0.0;
                }
                for (int t = 0; t < steps; t++) {
                    double* input_vec = convertState(flat[i]->states[t]);
                    double* p_new = forward(network, input_vec, temperature);

                    double* p_old = flat[i]->probs[t];

                    double kl_t = 0.0;
                    for (int k = 0; k < ACTION_COUNT; k++) {
                        double po = p_old[k];
                        if (po > 0.0) {
                            double pn = fmax(p_new[k], 1e-12);
                            kl_t += po * (log(po + 1e-12) - log(pn));
                        }
                    }
                    kl_sum += kl_t;

                    int aidx = actionToIndex(flat[i]->actions[t]);
                    if (aidx >= 0 && aidx < ACTION_COUNT) {
                        double po_a = fmax(p_old[aidx], 1e-12);
                        double pn_a = fmax(p_new[aidx], 1e-12);
                        double r = pn_a / po_a;
                        ratio_sum += r;
                        ratio_sq_sum += r * r;
                        if (r < ratio_min) ratio_min = r;
                        if (r > ratio_max) ratio_max = r;
                        if (r < (1.0 - clip_eps) || r > (1.0 + clip_eps)) clip_count++;
                        if (adv_flat) surr_sum += r * adv_flat[flat_idx];
                    }

                    flat_idx++;
                    free(input_vec);
                }
            }

            double mean_kl = kl_sum / (double)total_steps;
            double mean_ratio = ratio_sum / (double)total_steps;
            double var_ratio = (ratio_sq_sum / (double)total_steps) - (mean_ratio * mean_ratio);
            if (var_ratio < 0.0) var_ratio = 0.0;
            double std_ratio = sqrt(var_ratio);
            double clip_frac = (double)clip_count / (double)total_steps;
            double surrogate = (total_steps > 0) ? (surr_sum / (double)total_steps) : 0.0;

            double adv_mean = 0.0, adv_var = 0.0, adv_min = 0.0, adv_max = 0.0;
            if (adv_flat) {
                adv_min = DBL_MAX; adv_max = -DBL_MAX; adv_mean = 0.0;
                for (int i = 0; i < total_steps; i++) {
                    double a = adv_flat[i];
                    adv_mean += a;
                    if (a < adv_min) adv_min = a;
                    if (a > adv_max) adv_max = a;
                }
                adv_mean /= (double)total_steps;
                for (int i = 0; i < total_steps; i++) {
                    double d = adv_flat[i] - adv_mean;
                    adv_var += d * d;
                }
                adv_var /= (double)total_steps;
            }

            printf("  PPO diag  : KL=%-8.6f  ratio mean=%-7.4f std=%-7.4f min=%-7.4f max=%-7.4f  clip@%.2f=%.3f\n",
                   mean_kl, mean_ratio, std_ratio, ratio_min, ratio_max, clip_eps, clip_frac);
            if (adv_flat) {
                printf("            : surrogate(unclipped)=%-9.6f  adv mean=%-7.4f std=%-7.4f min=%-7.4f max=%-7.4f\n",
                       surrogate, adv_mean, sqrt(adv_var), adv_min, adv_max);
            } else {
                printf("            : surrogate(unclipped)=n/a  adv= n/a\n");
            }

            printf("            : value_loss=n/a  explained_var=n/a\n\n");
        } else {
            printf("  PPO diag  : n/a (no steps)\n\n");
        }

        if (adv_flat) free(adv_flat);
        if (A_per_traj) { 
            for (int b = 0; b < total_traj; b++) free(A_per_traj[b]);
            free(A_per_traj);
        }
        if (R_per_traj) { 
            for (int b = 0; b < total_traj; b++) free(R_per_traj[b]);
            free(R_per_traj);
        }

        free(traj_returns);

        episode++;
        saveLSTMCheckpoint(CHECKPOINT_PATH, network, (uint64_t)episode, (uint64_t)seed);

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
        free(b_temp);
    }

    freeLSTM(network);
    return 0;
}