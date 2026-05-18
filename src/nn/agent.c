#define _POSIX_C_SOURCE 200809L // Pour ftruncate

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Pour la parallélisation
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "agent.h"
#include "ppo.h"
#include "lstm.h"
#include "adam.h"
#include "config.h"
#include "reward.h"
#include "libretro_emu.h"

// Initialisation de l'agent
Agent* init_agent() {
    Agent* agent = malloc(sizeof(Agent));

    init_lstm(&agent->lstm, INPUT_SIZE, HIDDEN_SIZE);
    init_dense(&agent->policy_head, HIDDEN_SIZE, POLICY_OUTPUT_SIZE);
    init_dense(&agent->value_head, HIDDEN_SIZE, VALUE_OUTPUT_SIZE);

    return agent;
}

// Système d'entropie/température dynamique
void agent_set_schedule(Agent* agent, int epoch) {
    // Décroissance linéaire de l'entropie
    if (epoch >= ENTROPY_DECAY_EPOCHS) {
        agent->entropy_coeff = ENTROPY_MIN;
    } else {
        double frac = (double)epoch / (double)ENTROPY_DECAY_EPOCHS;
        agent->entropy_coeff = ENTROPY_INIT + frac * (ENTROPY_MIN - ENTROPY_INIT);
    }

    // TDécroissance linéaire de la température
    if (epoch >= TEMP_DECAY_EPOCHS) {
        agent->temperature = TEMP_MIN;
    } else {
        double frac_t = (double)epoch / (double)TEMP_DECAY_EPOCHS;
        agent->temperature = TEMP_INIT + frac_t * (TEMP_MIN - TEMP_INIT);
        if (agent->temperature < 1e-6) {
            agent->temperature = 1e-6;
        }
    }
}

// Fonction d'activation softmax (softmax = e^(x/ T ) / Σ(e^(x/ T ))) pour convertir les "logits" en distribution
void softmax(double* logits, int size, double* output, double temperature) {
    double max_logit = logits[0] / temperature;
    for (int i=1; i<size; i++) {
        double v = logits[i] / temperature;
        if (v > max_logit) {
            max_logit = v;
        }
    }

    double sum = 0;
    for (int i=0; i<size; i++) {
        sum += exp(logits[i] / temperature - max_logit); // On soustrait le maxmum pour de la stabilité numérique
    }
    for (int i=0; i<size; i++) {
        output[i] = exp(logits[i] / temperature - max_logit) / sum;
    }
}

int action_choice(double probs[POLICY_OUTPUT_SIZE]) {
    double r = (double)rand() / RAND_MAX;
    double s = 0;
    for (int i=0; i<POLICY_OUTPUT_SIZE; i++) {
        s += probs[i];
        if (r < s) {
            return i;
        }
    }
    return 0;
}

// Propagation pour tout l'agent
void agent_forward_t(Agent* agent, int state[INPUT_SIZE], Trajectory* traj, int t) {
    // Normalisation (norme 2) des 24 premières valeurs seulement
    int normalized_state[INPUT_SIZE];
    double sumsq = 0.0;
    for (int i = 0; i < 24; i++) {
        double v = (double)state[i];
        sumsq += v * v;
    }
    double norm = sqrt(sumsq) + 1e-8;
    for (int i = 0; i < 24; i++) {
        double v = (double)state[i] / norm;
        double scaled = v * 1000.0;
        normalized_state[i] = (int)scaled;
    }

    for (int i = 24; i < INPUT_SIZE; i++) {
        normalized_state[i] = state[i];
    }
    lstm_forward(&agent->lstm, traj, normalized_state, t);

    dense_forward(&agent->policy_head, agent->lstm.hidden_state, agent->policy_logits);
    dense_forward(&agent->value_head, agent->lstm.hidden_state, agent->value_logits);

    softmax(agent->policy_logits, agent->policy_head.output_size, traj->probs[t], agent->temperature);

    traj->actions[t] = action_choice(traj->probs[t]);
    traj->values[t] = agent->value_logits[0];

    if (is_done()) {
        traj->done[t] = 1; // Requis pour la GAE
    } else {
        traj->done[t] = 0;
    }

    memcpy(traj->states[t], state, sizeof(int) * INPUT_SIZE);
}

void agent_forward(Agent* agent, Trajectory* traj[NUM_ENVS]) {
    pid_t pids[NUM_ENVS];

    for (int env=0; env<NUM_ENVS; env++) {
        pid_t pid = fork();
        if (pid == 0) { // Processus enfant
            srand((unsigned int)env);
            // Création d'une instance de l'émulateur nécessaire après le fork (pour avoir des émulateurs différents)
            gba_set_core_silent(1); 
            gba_set_press_timing(3, 20);
            gba_set_video_enabled(0);
            gba_create("../libretro-super/dist/unix/mgba_libretro.so", "rom/pokemon.gba");
            gba_reset("rom/start.sav");

            char screen_path[64];
            snprintf(screen_path, sizeof(screen_path), "screenshots/screen_%d.bmp", env);

            for (int t=0; t<TRAJ_SIZE; t++) {
                int state[INPUT_SIZE];
                gba_state(state);

                if (t > 0) {
                    traj[env]->rewards[t-1] = reward(traj[env]->states[t-1], state);
                }

                agent_forward_t(agent, state, traj[env], t);

                gba_button(traj[env]->actions[t]);
                // gba_screen(screen_path); // Attention, gourmand en ressources, à n'utiliser que pour le debug
            }

            int final_state[INPUT_SIZE];
            gba_state(final_state);
            traj[env]->rewards[TRAJ_SIZE-1] = reward(traj[env]->states[TRAJ_SIZE-1], final_state);

            double advantages[TRAJ_SIZE];
            compute_advantages(traj[env]->rewards, traj[env]->values, traj[env]->done, advantages);
            memcpy(traj[env]->advantages, advantages, TRAJ_SIZE * sizeof(double));

            gba_destroy();
            exit(0);
        } else { // Processus parent
            pids[env] = pid;
        }
    }

    // Attendre la fin de tous les processus enfants
    for (int env=0; env<NUM_ENVS; env++) {
        waitpid(pids[env], NULL, 0);
    }
}

// Rétropropagation pour tout l'agent
void recompute_probs(Agent* agent, Minibatch* current_minibatch, double new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE]) {
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        dense_forward(&agent->policy_head, current_minibatch->hidden_states[t], agent->policy_logits);
        softmax(agent->policy_logits, agent->policy_head.output_size, new_probs[t], agent->temperature);
    }
}

double compute_std(double array[MINIBATCH_SIZE], double mean) {
    double variance = 0;

    for (int i=0; i<MINIBATCH_SIZE; i++) {
        double diff = array[i] - mean;
        variance += diff * diff;
    }
    variance /= MINIBATCH_SIZE;

    return sqrt(variance);
}

void print_epoch_summary(const EpochSummary* summary) {
    int total_steps = summary->minibatch_count * MINIBATCH_SIZE;
    double ppo_loss = summary->ppo_loss_sum / summary->minibatch_count;
    double value_loss = summary->value_loss_sum / summary->minibatch_count;
    double value_mean = summary->value_sum / total_steps;
    double value_std = sqrt(fmax(0.0, (summary->value_sq_sum / total_steps) - value_mean * value_mean));
    double adv_mean = summary->adv_sum / total_steps;
    double adv_std = sqrt(fmax(0.0, (summary->adv_sq_sum / total_steps) - adv_mean * adv_mean));
    double ratio_mean = summary->ratio_sum / total_steps;
    double unclipped_ratio_mean = summary->unclipped_ratio_sum / total_steps;
    double clipped_percentage = (summary->clipped_count / total_steps) * 100.0;
    double entropy_mean = summary->entropy_sum / total_steps;
    double kl_mean = summary->kl_sum / total_steps;

    printf("EPOCH SUMMARY: PPO loss: %.6f | Value loss: %.6f | Average Reward: %.4f | Value mean: %.4f | Value std: %.4f | Adv mean: %.4f | Adv std: %.4f | Ratio mean: %.4f | Unclipped mean: %.4f | Clipped %%: %.2f%% | Entropy: %.4f | KL: %.6f\n",
        ppo_loss, value_loss, (summary->reward_sum/((double)(summary->minibatch_count * MINIBATCH_SIZE))), value_mean, value_std, adv_mean, adv_std, ratio_mean, unclipped_ratio_mean, clipped_percentage, entropy_mean, kl_mean);

    printf("Action distribution (%%): ");
    for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
        double pct = (summary->action_counts[j] / total_steps) * 100.0;
        printf("A%d: %.2f%% ", j, pct);
    }
    printf("\n");
}

double agent_backward_minibatch(Agent* agent, Optimizer* optim, Minibatch* minibatch) {
    // Rétropropagation de la value head/critic
    double returns[MINIBATCH_SIZE];
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        returns[t] = minibatch->advantages[t] + minibatch->values[t]; // R_t = A_t + V_t (les returns correspondent aux récompenses sur le long terme, c'est-à-dire la somme pondérées des récompenses futures)
    }

    double dL_dw_v[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    double dL_db_v[MAX_OUTPUT_SIZE] = {0};
    double dL_dinput_v[MINIBATCH_SIZE][HIDDEN_SIZE] = {0};
    minibatch->value_loss = value_backward(&agent->value_head, minibatch->values, returns, minibatch->hidden_states, dL_dw_v, dL_db_v, dL_dinput_v);

    // Rétropropagation de la policy head/actor
    double new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE];
    recompute_probs(agent, minibatch, new_probs);

    double dL_dw_p[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    double dL_db_p[MAX_OUTPUT_SIZE] = {0};
    double dL_dinput_p[MINIBATCH_SIZE][HIDDEN_SIZE] = {0};
    policy_backward(&agent->policy_head, minibatch, new_probs, dL_dw_p, dL_db_p, dL_dinput_p, agent->entropy_coeff);

    // Rétropropagation à travers le LSTM
    double dL_dwf[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwi[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwc[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwo[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dbf[HIDDEN_SIZE] = {0};
    double dL_dbi[HIDDEN_SIZE] = {0};
    double dL_dbc[HIDDEN_SIZE] = {0};
    double dL_dbo[HIDDEN_SIZE] = {0};

    lstm_backward(&agent->lstm, minibatch, dL_dinput_v, dL_dinput_p, minibatch->c_ini, dL_dwf, dL_dwi, dL_dwc, dL_dwo, dL_dbf, dL_dbi, dL_dbc, dL_dbo);

    // Mise à jour des poids avec Adam
    optim->t += 1;

    optimizer_step_matrix_1(optim, agent->value_head.w, agent->value_head.w_m, agent->value_head.w_v, dL_dw_v, agent->value_head.output_size, agent->value_head.input_size);
    optimizer_step_vector_1(optim, agent->value_head.b, agent->value_head.b_m, agent->value_head.b_v, dL_db_v, agent->value_head.output_size);

    optimizer_step_matrix_1(optim, agent->policy_head.w, agent->policy_head.w_m, agent->policy_head.w_v, dL_dw_p, agent->policy_head.output_size, agent->policy_head.input_size);
    optimizer_step_vector_1(optim, agent->policy_head.b, agent->policy_head.b_m, agent->policy_head.b_v, dL_db_p, agent->policy_head.output_size);

    // Mise à jour des poids du LSTM avec Adam
    optimizer_step_matrix_2(optim, agent->lstm.wf, agent->lstm.wf_m, agent->lstm.wf_v, dL_dwf, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wi, agent->lstm.wi_m, agent->lstm.wi_v, dL_dwi, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wc, agent->lstm.wc_m, agent->lstm.wc_v, dL_dwc, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wo, agent->lstm.wo_m, agent->lstm.wo_v, dL_dwo, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bf, agent->lstm.bf_m, agent->lstm.bf_v, dL_dbf, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bi, agent->lstm.bi_m, agent->lstm.bi_v, dL_dbi, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bc, agent->lstm.bc_m, agent->lstm.bc_v, dL_dbc, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bo, agent->lstm.bo_m, agent->lstm.bo_v, dL_dbo, HIDDEN_SIZE);


    double kl_mean = 0.0;
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        kl_mean += minibatch->kl[t];
    }
    kl_mean /= MINIBATCH_SIZE;

    return kl_mean;
}

double agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj[NUM_ENVS], EpochSummary* summary) {
    int num_minibatches = BATCH_SIZE / MINIBATCH_SIZE;
    Minibatch* minibatches[num_minibatches];

    // Création et mélange des minibatchs
    int idx[num_minibatches];
    for (int i=0; i<num_minibatches; i++) {
        idx[i] = i;
    }

    for (int i=0; i<num_minibatches; i++) {
        int j = i + rand() / (RAND_MAX / (num_minibatches - i) + 1);
        int t = idx[j];
        idx[j] = idx[i];
        idx[i] = t;
    }

    for (int i=0; i<num_minibatches; i++) {
        int num_traj = idx[i] / (TRAJ_SIZE / MINIBATCH_SIZE);
        int num_traj_minibatch = idx[i] % (TRAJ_SIZE / MINIBATCH_SIZE);
        int start = num_traj_minibatch * MINIBATCH_SIZE;

        minibatches[i] = malloc(sizeof(Minibatch));
        memcpy(minibatches[i]->states, &traj[num_traj]->states[start], sizeof(minibatches[i]->states));
        memcpy(minibatches[i]->hidden_states, &traj[num_traj]->hidden_states[start], sizeof(minibatches[i]->hidden_states));
        memcpy(minibatches[i]->z, &traj[num_traj]->z[start], sizeof(minibatches[i]->z));
        memcpy(minibatches[i]->f, &traj[num_traj]->f[start], sizeof(minibatches[i]->f));
        memcpy(minibatches[i]->i, &traj[num_traj]->i[start], sizeof(minibatches[i]->i));
        memcpy(minibatches[i]->g, &traj[num_traj]->g[start], sizeof(minibatches[i]->g));
        memcpy(minibatches[i]->o, &traj[num_traj]->o[start], sizeof(minibatches[i]->o));
        memcpy(minibatches[i]->c, &traj[num_traj]->c[start], sizeof(minibatches[i]->c));
        // On trouve la bonne cellule au début du minibatch
        if (start > 0 && traj[num_traj]->done[start-1] == 0) {
            memcpy(minibatches[i]->c_ini, traj[num_traj]->c[start-1], sizeof(minibatches[i]->c_ini));
        } else { // Sinon on initialise la cellule à 0
            for (int j=0; j<HIDDEN_SIZE; j++) {
                minibatches[i]->c_ini[j] = 0.0;
            }
        }
        memcpy(minibatches[i]->actions, &traj[num_traj]->actions[start], sizeof(minibatches[i]->actions));
        memcpy(minibatches[i]->rewards, &traj[num_traj]->rewards[start], sizeof(minibatches[i]->rewards));
        memcpy(minibatches[i]->probs, &traj[num_traj]->probs[start], sizeof(minibatches[i]->probs));
        memcpy(minibatches[i]->values, &traj[num_traj]->values[start], sizeof(minibatches[i]->values));
        memcpy(minibatches[i]->advantages, &traj[num_traj]->advantages[start], sizeof(minibatches[i]->advantages));
        memcpy(minibatches[i]->done, &traj[num_traj]->done[start], sizeof(minibatches[i]->done));
    }

    // Rétropropagation
    double epoch_kl_sum = 0.0;
    for (int i=0; i<num_minibatches; i++) {
        double minibatch_kl = agent_backward_minibatch(agent, optim, minibatches[i]);
        epoch_kl_sum += minibatch_kl;
    }
    double mean_epoch_kl = epoch_kl_sum / num_minibatches;

    for (int i=0; i<num_minibatches; i++) {
        Minibatch* mb = minibatches[i];
        summary->ppo_loss_sum += mb->ppo_loss;
        summary->value_loss_sum += mb->value_loss;
        for (int t=0; t<MINIBATCH_SIZE; t++) {
            summary->reward_sum += mb->rewards[t];
            summary->value_sum += mb->values[t];
            summary->value_sq_sum += mb->values[t] * mb->values[t];
            summary->adv_sum += mb->advantages[t];
            summary->adv_sq_sum += mb->advantages[t] * mb->advantages[t];
            summary->ratio_sum += mb->ratios[t];
            summary->unclipped_ratio_sum += mb->unclipped_ratios[t];
            summary->clipped_count += mb->clipped[t];
            for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
                double p = mb->probs[t][j];
                summary->entropy_sum += -p * log(p + 1e-10);
            }
            summary->kl_sum += mb->kl[t];
            summary->action_counts[mb->actions[t]] += 1.0;
        }
    }
    summary->minibatch_count += num_minibatches;

    // Libération de la mémoire
    for (int i=0; i<num_minibatches; i++) {
        free(minibatches[i]);
    }

    return mean_epoch_kl;
}

void train_epoch(Agent* agent, Optimizer* optim) {
    Trajectory* traj[NUM_ENVS];
    for (int env=0; env<NUM_ENVS; env++) {
        // Allocation de la mémoire partagée (nom unique par worker)
        char shm_name[64];
        snprintf(shm_name, sizeof(shm_name), "/worker_%d", env);
        
        // Nettoie d'éventuelles mémoires partagées résiduelles de runs précédents
        shm_unlink(shm_name);
        
        // Crée une mémoire partagée neuve
        int shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
        if (shm_fd < 0) {
            perror("shm_open");
            exit(1);
        }
        if (ftruncate(shm_fd, sizeof(Trajectory)) != 0) {
            perror("ftruncate");
            close(shm_fd);
            shm_unlink(shm_name);
            exit(1);
        }

        traj[env] = mmap(NULL, sizeof(Trajectory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (traj[env] == MAP_FAILED) {
            perror("mmap");
            close(shm_fd);
            shm_unlink(shm_name);
            exit(1);
        }
        close(shm_fd);
    }

    agent_forward(agent, traj);

    EpochSummary epoch_summary = {0};

    for (int ppo_epoch=0; ppo_epoch<PPO_EPOCHS; ppo_epoch++) {
        double mean_kl = agent_backward(agent, optim, traj, &epoch_summary);
        if (ppo_epoch + 1 >= KL_MIN_EPOCHS && mean_kl > KL_TARGET) {
            break;
        }
    }

    print_epoch_summary(&epoch_summary);

    // Libération de la mémoire partagée
    for (int env=0; env<NUM_ENVS; env++) {
        char shm_name[64];
        snprintf(shm_name, sizeof(shm_name), "/worker_%d", env);
        munmap(traj[env], sizeof(Trajectory));
        shm_unlink(shm_name);
    }
}

void train(Agent* agent, Optimizer* optim, int epochs) {
    for (int epoch=0; epoch<epochs; epoch++) {
        reset_flags();
        printf("\n========== Epoch %d/%d ==========\n", epoch+1, epochs);
        agent_set_schedule(agent, epoch);
        printf("Entropy coeff: %.6f | Temperature: %.6f\n", agent->entropy_coeff, agent->temperature);
        train_epoch(agent, optim);
    }
}

int main() {
    Agent* agent = init_agent();
    Optimizer* optim = malloc(sizeof(Optimizer));
    optim->lr = LEARNING_RATE;
    optim->beta1 = BETA1;
    optim->beta2 = BETA2;
    optim->epsilon = EPSILON_ADAM;
    optim->t = 0;

    train(agent, optim, EPOCHS);

    return 0;
}