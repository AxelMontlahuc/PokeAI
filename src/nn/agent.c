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

// Fonction d'activation softmax (softmax = e^x / Σ(e^x)) pour convertir les "logits" (sorties brutes de la politique) en une distribution de probabilité
void softmax(double* logits, int size, double* output) {
    double max_logit = logits[0];
    for (int i=1; i<size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum = 0;
    for (int i=0; i<size; i++) {
        sum += exp(logits[i] - max_logit); // On soustrait le maximum pour des questions de stabilité numérique
    }
    for (int i=0; i<size; i++) {
        output[i] = exp(logits[i] - max_logit) / sum;
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
    lstm_forward(&agent->lstm, traj, state, t);

    dense_forward(&agent->policy_head, agent->lstm.hidden_state, agent->policy_logits);
    dense_forward(&agent->value_head, agent->lstm.hidden_state, agent->value_logits);

    softmax(agent->policy_logits, agent->policy_head.output_size, traj->probs[t]);

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
            gba_set_press_timing(6, 20);
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
        softmax(agent->policy_logits, agent->policy_head.output_size, new_probs[t]);
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

void logging(Minibatch* minibatch) {
    double loss_ppo = minibatch->ppo_loss;
    double loss_value = minibatch->value_loss;
    double reward_sum = 0;
    double value_mean = 0;
    double value_std = 0;
    double advantage_mean = 0;
    double advantage_std = 0;
    double ratio_mean = 0;
    double unclipped_ratio_mean = 0;
    double clipped_percentage = 0;
    double mean_entropy = 0;
    double kl_mean = 0;
    double actions[POLICY_OUTPUT_SIZE] = {0};

    for (int t=0; t<MINIBATCH_SIZE; t++) {
        reward_sum += minibatch->rewards[t];
        value_mean += minibatch->values[t];
        advantage_mean += minibatch->advantages[t];
        ratio_mean += minibatch->ratios[t];
        unclipped_ratio_mean += minibatch->unclipped_ratios[t];
        clipped_percentage += minibatch->clipped[t];

        double current_entropy = 0;
        for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
            current_entropy -= minibatch->probs[t][j] * log(minibatch->probs[t][j] + 1e-10); // On évite log(0)
        }
        mean_entropy += current_entropy;
        kl_mean += minibatch->kl[t];
        actions[minibatch->actions[t]] += 1;
    }

    value_mean /= MINIBATCH_SIZE;
    advantage_mean /= MINIBATCH_SIZE;
    value_std = compute_std(minibatch->values, value_mean);
    advantage_std = compute_std(minibatch->advantages, advantage_mean);
    ratio_mean /= MINIBATCH_SIZE;
    unclipped_ratio_mean /= MINIBATCH_SIZE;
    clipped_percentage /= MINIBATCH_SIZE;
    clipped_percentage *= 100;
    kl_mean /= MINIBATCH_SIZE;
    mean_entropy /= MINIBATCH_SIZE;

    for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
        actions[j] /= MINIBATCH_SIZE;
        actions[j] *= 100;
    }

    printf("Reward sum: %.4f | Loss: %.4f | Value mean: %.4f | Value std: %.4f | Value loss: %.4f | Advantage mean: %.4f | Advantage std: %.4f | Ratio mean: %.4f | Unclipped ratio mean: %.4f | Clipped percentage: %.2f%% | Mean entropy: %.4f | KL divergence: %.4f\n", reward_sum, loss_ppo, value_mean, value_std, loss_value, advantage_mean, advantage_std, ratio_mean, unclipped_ratio_mean, clipped_percentage, mean_entropy, kl_mean);
    printf("Action distribution: ");
    for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
        printf("Action %d: %.2f%%, ", j, actions[j]);
    }
    printf("\n\n");
}

void agent_backward_minibatch(Agent* agent, Optimizer* optim, Minibatch* minibatch) {
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
    policy_backward(&agent->policy_head, minibatch, new_probs, dL_dw_p, dL_db_p, dL_dinput_p);

    // Rétropropagation à travers le LSTM
    double dL_dwf[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwi[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwc[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwo[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dbf[HIDDEN_SIZE] = {0};
    double dL_dbi[HIDDEN_SIZE] = {0};
    double dL_dbc[HIDDEN_SIZE] = {0};
    double dL_dbo[HIDDEN_SIZE] = {0};

    double c_ini[HIDDEN_SIZE] = {0}; // On suppose que la mémoire à long terme est initialisée à zéro au début de chaque trajectoire pour l'instant, mais il faudra à terme le stocker et le faire passer d'une trajectoire à l'autre
    lstm_backward(&agent->lstm, minibatch, dL_dinput_v, dL_dinput_p, c_ini, dL_dwf, dL_dwi, dL_dwc, dL_dwo, dL_dbf, dL_dbi, dL_dbc, dL_dbo);

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

    logging(minibatch);
}

void agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj[NUM_ENVS]) {
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
        memcpy(minibatches[i]->actions, &traj[num_traj]->actions[start], sizeof(minibatches[i]->actions));
        memcpy(minibatches[i]->rewards, &traj[num_traj]->rewards[start], sizeof(minibatches[i]->rewards));
        memcpy(minibatches[i]->probs, &traj[num_traj]->probs[start], sizeof(minibatches[i]->probs));
        memcpy(minibatches[i]->values, &traj[num_traj]->values[start], sizeof(minibatches[i]->values));
        memcpy(minibatches[i]->advantages, &traj[num_traj]->advantages[start], sizeof(minibatches[i]->advantages));
        memcpy(minibatches[i]->done, &traj[num_traj]->done[start], sizeof(minibatches[i]->done));
    }

    // Rétropropagation
    for (int i=0; i<num_minibatches; i++) {
        agent_backward_minibatch(agent, optim, minibatches[i]);
    }

    // Libération de la mémoire
    for (int i=0; i<num_minibatches; i++) {
        free(minibatches[i]);
    }
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

    for (int ppo_epoch=0; ppo_epoch<PPO_EPOCHS; ppo_epoch++) {
        printf("PPO Epoch %d/%d\n", ppo_epoch+1, PPO_EPOCHS);
        agent_backward(agent, optim, traj);
    }

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