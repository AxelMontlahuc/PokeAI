#define _POSIX_C_SOURCE 200809L // Pour ftruncate

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Pour la parallélisation
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include "agent.h"
#include "ppo.h"
#include "lstm.h"
#include "adam.h"
#include "config.h"
#include "reward.h"
#include "libretro_emu.h"
#include "checkpoint.h"

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
        float frac = (float)epoch / (float)ENTROPY_DECAY_EPOCHS;
        agent->entropy_coeff = ENTROPY_INIT + frac * (ENTROPY_MIN - ENTROPY_INIT);
    }

    // TDécroissance linéaire de la température
    if (epoch >= TEMP_DECAY_EPOCHS) {
        agent->temperature = TEMP_MIN;
    } else {
        float frac_t = (float)epoch / (float)TEMP_DECAY_EPOCHS;
        agent->temperature = TEMP_INIT + frac_t * (TEMP_MIN - TEMP_INIT);
        if (agent->temperature < 1e-6f) {
            agent->temperature = 1e-6f;
        }
    }
}

// Fonction d'activation softmax (softmax = e^(x/ T ) / Σ(e^(x/ T ))) pour convertir les "logits" en distribution
void softmax(float* logits, int size, float* output, float temperature) {
    float max_logit = logits[0] / temperature;
    for (int i=1; i<size; i++) {
        float v = logits[i] / temperature;
        if (v > max_logit) {
            max_logit = v;
        }
    }

    float sum = 0.0f;
    for (int i=0; i<size; i++) {
        sum += expf(logits[i] / temperature - max_logit); // On soustrait le maxmum pour de la stabilité numérique
    }
    for (int i=0; i<size; i++) {
        output[i] = expf(logits[i] / temperature - max_logit) / sum;
    }
}

int action_choice(float probs[POLICY_OUTPUT_SIZE]) {
    float r = (float)rand() / (float)RAND_MAX;
    float s = 0.0f;
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
    float sumsq = 0.0f;
    for (int i = 0; i < 24; i++) {
        float v = (float)state[i];
        sumsq += v * v;
    }
    float norm = sqrtf(sumsq) + 1e-8f;
    for (int i = 0; i < 24; i++) {
        float v = (float)state[i] / norm;
        float scaled = v * 1000.0f;
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
            gba_set_press_timing(1, 20);
            gba_set_video_enabled(0);
            gba_set_frameskip(0);
            gba_create("../libretro-super/libretro-gpsp/gpsp_libretro.so", "rom/pokemon.gba");
            gba_reset("rom/start_gpsp.sav");

            char screen_path[64];
            snprintf(screen_path, sizeof(screen_path), "screenshots/screen_%d.bmp", env);

            // Reinitialisation de la memoire recurrente du LSTM pour la nouvelle trajectoire
            memset(agent->lstm.hidden_state, 0, sizeof(agent->lstm.hidden_state));
            memset(agent->lstm.cell_state, 0, sizeof(agent->lstm.cell_state));

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

            float advantages[TRAJ_SIZE];
            compute_advantages(traj[env]->rewards, traj[env]->values, traj[env]->done, advantages);
            memcpy(traj[env]->advantages, advantages, TRAJ_SIZE * sizeof(float));

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
void recompute_probs(Agent* agent, Minibatch* current_minibatch, float new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE]) {
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        dense_forward(&agent->policy_head, current_minibatch->hidden_states[t], agent->policy_logits);
        softmax(agent->policy_logits, agent->policy_head.output_size, new_probs[t], agent->temperature);
    }
}

void recompute_values(Agent* agent, Minibatch* current_minibatch, float new_values[MINIBATCH_SIZE]) {
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        float value_logits[VALUE_OUTPUT_SIZE];
        dense_forward(&agent->value_head, current_minibatch->hidden_states[t], value_logits);
        new_values[t] = value_logits[0];
    }
}

float compute_std(float array[MINIBATCH_SIZE], float mean) {
    float variance = 0.0f;

    for (int i=0; i<MINIBATCH_SIZE; i++) {
        float diff = array[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)MINIBATCH_SIZE;

    return sqrtf(variance);
}

void print_epoch_summary(EpochSummary* summary) {
    int total_steps = summary->minibatch_count * MINIBATCH_SIZE;
    float ppo_loss = summary->ppo_loss_sum / (float)summary->minibatch_count;
    float value_loss = summary->value_loss_sum / (float)summary->minibatch_count;
    float value_mean = summary->value_sum / (float)total_steps;
    float value_std = sqrtf(fmaxf(0.0f, (summary->value_sq_sum / (float)total_steps) - value_mean * value_mean));
    float adv_mean = summary->adv_sum / (float)total_steps;
    float adv_std = sqrtf(fmaxf(0.0f, (summary->adv_sq_sum / (float)total_steps) - adv_mean * adv_mean));
    float ratio_mean = summary->ratio_sum / (float)total_steps;
    float unclipped_ratio_mean = summary->unclipped_ratio_sum / (float)total_steps;
    float clipped_percentage = (summary->clipped_count / (float)total_steps) * 100.0f;
    float entropy_mean = summary->entropy_sum / (float)total_steps;
    float kl_mean = summary->kl_sum / (float)total_steps;

    printf("EPOCH SUMMARY: PPO loss: %.6f | Value loss: %.6f | Average Reward (per env): %.4f | Value mean: %.4f | Value std: %.4f | Adv mean: %.4f | Adv std: %.4f | Ratio mean: %.4f | Unclipped mean: %.4f | Clipped %%: %.2f%% | Entropy: %.4f | KL: %.6f\n",
        ppo_loss, value_loss, (summary->reward_sum/((float)(NUM_ENVS))), value_mean, value_std, adv_mean, adv_std, ratio_mean, unclipped_ratio_mean, clipped_percentage, entropy_mean, kl_mean);

    printf("Action distribution (%%): ");
    for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
        float pct = (summary->action_counts[j] / (float)total_steps) * 100.0f;
        printf("A%d: %.2f%% ", j, pct);
    }
    printf("\n");
}

float agent_backward_minibatch(Agent* agent, Optimizer* optim, Minibatch* minibatch) {
    lstm_recompute_minibatch(&agent->lstm, minibatch);
    // Rétropropagation de la value head/critic
    float returns[MINIBATCH_SIZE];
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        returns[t] = minibatch->advantages[t] + minibatch->values[t]; // R_t = A_t + V_t (les returns correspondent aux récompenses sur le long terme, c'est-à-dire la somme pondérées des récompenses futures)
    }

    float dL_dw_v[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    float dL_db_v[MAX_OUTPUT_SIZE] = {0};
    float dL_dinput_v[MINIBATCH_SIZE][HIDDEN_SIZE] = {0};
    float new_values[MINIBATCH_SIZE];
    recompute_values(agent, minibatch, new_values);
    minibatch->value_loss = value_backward(&agent->value_head, new_values, returns, minibatch->hidden_states, dL_dw_v, dL_db_v, dL_dinput_v);
    // On partage le gradient en deux (une moitié pour le critic, une moitié pour l'actor)
    for (int i = 0; i < agent->value_head.output_size; i++) {
        for (int j = 0; j < agent->value_head.input_size; j++) {
            dL_dw_v[i][j] *= 0.5f;
        }
        dL_db_v[i] *= 0.5f;
    }
    for (int t = 0; t < MINIBATCH_SIZE; t++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            dL_dinput_v[t][j] *= 0.5f;
        }
    }
    memcpy(minibatch->values, new_values, sizeof(float) * MINIBATCH_SIZE);

    // Rétropropagation de la policy head/actor
    float new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE];
    recompute_probs(agent, minibatch, new_probs);

    float dL_dw_p[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    float dL_db_p[MAX_OUTPUT_SIZE] = {0};
    float dL_dinput_p[MINIBATCH_SIZE][HIDDEN_SIZE] = {0};
    policy_backward(&agent->policy_head, minibatch, new_probs, dL_dw_p, dL_db_p, dL_dinput_p, agent->entropy_coeff, agent->temperature);

    // Rétropropagation à travers le LSTM
    float dL_dwf[HIDDEN_SIZE][COL_SIZE] = {0};
    float dL_dwi[HIDDEN_SIZE][COL_SIZE] = {0};
    float dL_dwc[HIDDEN_SIZE][COL_SIZE] = {0};
    float dL_dwo[HIDDEN_SIZE][COL_SIZE] = {0};
    float dL_dbf[HIDDEN_SIZE] = {0};
    float dL_dbi[HIDDEN_SIZE] = {0};
    float dL_dbc[HIDDEN_SIZE] = {0};
    float dL_dbo[HIDDEN_SIZE] = {0};

    lstm_backward(&agent->lstm, minibatch, dL_dinput_v, dL_dinput_p, minibatch->c_ini, dL_dwf, dL_dwi, dL_dwc, dL_dwo, dL_dbf, dL_dbi, dL_dbc, dL_dbo);

    // Clipping des gradients pour éviter qu'ils explosent (indépendamment pour le critic et l'actor pour que l'un ne prédomine pas sur l'autre)
    float policy_grad_norm_sq = 0.0f;
    for (int i = 0; i < agent->policy_head.output_size; i++) {
        for (int j = 0; j < agent->policy_head.input_size; j++) {
            policy_grad_norm_sq += dL_dw_p[i][j] * dL_dw_p[i][j];
        }
        policy_grad_norm_sq += dL_db_p[i] * dL_db_p[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < COL_SIZE; j++) {
            policy_grad_norm_sq += dL_dwf[i][j] * dL_dwf[i][j];
            policy_grad_norm_sq += dL_dwi[i][j] * dL_dwi[i][j];
            policy_grad_norm_sq += dL_dwc[i][j] * dL_dwc[i][j];
            policy_grad_norm_sq += dL_dwo[i][j] * dL_dwo[i][j];
        }
        policy_grad_norm_sq += dL_dbf[i] * dL_dbf[i];
        policy_grad_norm_sq += dL_dbi[i] * dL_dbi[i];
        policy_grad_norm_sq += dL_dbc[i] * dL_dbc[i];
        policy_grad_norm_sq += dL_dbo[i] * dL_dbo[i];
    }
    float policy_grad_norm = sqrtf(policy_grad_norm_sq);
    if (policy_grad_norm > MAX_GRAD_NORM) {
        float scale = MAX_GRAD_NORM / (policy_grad_norm + 1e-8f);
        for (int i = 0; i < agent->policy_head.output_size; i++) {
            for (int j = 0; j < agent->policy_head.input_size; j++) {
                dL_dw_p[i][j] *= scale;
            }
            dL_db_p[i] *= scale;
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < COL_SIZE; j++) {
                dL_dwf[i][j] *= scale;
                dL_dwi[i][j] *= scale;
                dL_dwc[i][j] *= scale;
                dL_dwo[i][j] *= scale;
            }
            dL_dbf[i] *= scale;
            dL_dbi[i] *= scale;
            dL_dbc[i] *= scale;
            dL_dbo[i] *= scale;
        }
    }

    float value_grad_norm_sq = 0.0f;
    for (int i = 0; i < agent->value_head.output_size; i++) {
        for (int j = 0; j < agent->value_head.input_size; j++) {
            value_grad_norm_sq += dL_dw_v[i][j] * dL_dw_v[i][j];
        }
        value_grad_norm_sq += dL_db_v[i] * dL_db_v[i];
    }
    float value_grad_norm = sqrtf(value_grad_norm_sq);
    if (value_grad_norm > MAX_GRAD_NORM) {
        float scale = MAX_GRAD_NORM / (value_grad_norm + 1e-8f);
        for (int i = 0; i < agent->value_head.output_size; i++) {
            for (int j = 0; j < agent->value_head.input_size; j++) {
                dL_dw_v[i][j] *= scale;
            }
            dL_db_v[i] *= scale;
        }
    }

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


    float kl_mean = 0.0f;
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        kl_mean += minibatch->kl[t];
    }
    kl_mean /= (float)MINIBATCH_SIZE;

    return kl_mean;
}

float agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj[NUM_ENVS], EpochSummary* summary) {
    int num_minibatches = BATCH_SIZE / MINIBATCH_SIZE;
    Minibatch* minibatches[num_minibatches];

    // Création et mélange des minibatchs
    int total_seqs = NUM_ENVS * (TRAJ_SIZE / SEQ_LEN);
    int seq_idx[total_seqs];
    for (int i=0; i<total_seqs; i++) {
        seq_idx[i] = i;
    }

    for (int i=0; i<total_seqs; i++) {
        int range = total_seqs - i;
        int j = i + (rand() % range);
        int t = seq_idx[j];
        seq_idx[j] = seq_idx[i];
        seq_idx[i] = t;
    }

    for (int i=0; i<num_minibatches; i++) {
        minibatches[i] = malloc(sizeof(Minibatch));
        for (int s=0; s<NUM_SEQS; s++) {
            int current_seq_idx = seq_idx[i * NUM_SEQS + s];
            int num_traj = current_seq_idx / (TRAJ_SIZE / SEQ_LEN);
            int seq_in_env = current_seq_idx % (TRAJ_SIZE / SEQ_LEN);
            int start = seq_in_env * SEQ_LEN;
            int mb_offset = s * SEQ_LEN;

            memcpy(&minibatches[i]->states[mb_offset], &traj[num_traj]->states[start], sizeof(int) * SEQ_LEN * INPUT_SIZE);
            memcpy(&minibatches[i]->hidden_states[mb_offset], &traj[num_traj]->hidden_states[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);
            memcpy(&minibatches[i]->z[mb_offset], &traj[num_traj]->z[start], sizeof(float) * SEQ_LEN * COL_SIZE);
            memcpy(&minibatches[i]->f[mb_offset], &traj[num_traj]->f[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);
            memcpy(&minibatches[i]->i[mb_offset], &traj[num_traj]->i[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);
            memcpy(&minibatches[i]->g[mb_offset], &traj[num_traj]->g[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);
            memcpy(&minibatches[i]->o[mb_offset], &traj[num_traj]->o[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);
            memcpy(&minibatches[i]->c[mb_offset], &traj[num_traj]->c[start], sizeof(float) * SEQ_LEN * HIDDEN_SIZE);

            if (start > 0 && traj[num_traj]->done[start-1] == 0) {
                memcpy(minibatches[i]->c_ini[s], traj[num_traj]->c[start-1], sizeof(float) * HIDDEN_SIZE);
                memcpy(minibatches[i]->h_ini[s], traj[num_traj]->hidden_states[start-1], sizeof(float) * HIDDEN_SIZE);
            } else {
                for (int j=0; j<HIDDEN_SIZE; j++) {
                    minibatches[i]->c_ini[s][j] = 0.0f;
                    minibatches[i]->h_ini[s][j] = 0.0f;
                }
            }

            memcpy(&minibatches[i]->actions[mb_offset], &traj[num_traj]->actions[start], sizeof(int) * SEQ_LEN);
            memcpy(&minibatches[i]->rewards[mb_offset], &traj[num_traj]->rewards[start], sizeof(float) * SEQ_LEN);
            memcpy(&minibatches[i]->probs[mb_offset], &traj[num_traj]->probs[start], sizeof(float) * SEQ_LEN * POLICY_OUTPUT_SIZE);
            memcpy(&minibatches[i]->values[mb_offset], &traj[num_traj]->values[start], sizeof(float) * SEQ_LEN);
            memcpy(&minibatches[i]->advantages[mb_offset], &traj[num_traj]->advantages[start], sizeof(float) * SEQ_LEN);
            memcpy(&minibatches[i]->done[mb_offset], &traj[num_traj]->done[start], sizeof(int) * SEQ_LEN);
        }
    }

    // Rétropropagation
    float epoch_kl_sum = 0.0f;
    for (int i=0; i<num_minibatches; i++) {
        float minibatch_kl = agent_backward_minibatch(agent, optim, minibatches[i]);
        epoch_kl_sum += minibatch_kl;
    }
    float mean_epoch_kl = epoch_kl_sum / (float)num_minibatches;

    for (int i=0; i<num_minibatches; i++) {
        Minibatch* mb = minibatches[i];
        summary->ppo_loss_sum += mb->ppo_loss;
        summary->value_loss_sum += mb->value_loss;
        for (int t=0; t<MINIBATCH_SIZE; t++) {
            summary->value_sum += mb->values[t];
            summary->value_sq_sum += mb->values[t] * mb->values[t];
            summary->adv_sum += mb->advantages[t];
            summary->adv_sq_sum += mb->advantages[t] * mb->advantages[t];
            summary->ratio_sum += mb->ratios[t];
            summary->unclipped_ratio_sum += mb->unclipped_ratios[t];
            summary->clipped_count += (float)(mb->clipped[t]);
            for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
                float p = mb->probs[t][j];
                summary->entropy_sum += -p * logf(p + 1e-10f);
            }
            summary->kl_sum += mb->kl[t];
            summary->action_counts[mb->actions[t]] += 1.0f;
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

    // On accumule les recompenses de toutes les trajectoires une seule fois pour eviter les repetitions
    float epoch_reward_sum = 0.0f;
    for (int env=0; env<NUM_ENVS; env++) {
        for (int t=0; t<TRAJ_SIZE; t++) {
            epoch_reward_sum += traj[env]->rewards[t];
        }
    }
    epoch_summary.reward_sum = epoch_reward_sum;

    for (int ppo_epoch=0; ppo_epoch<PPO_EPOCHS; ppo_epoch++) {
        float mean_kl = agent_backward(agent, optim, traj, &epoch_summary);
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

void train(Agent* agent, Optimizer* optim, int start_epoch, int epochs) {
    mkdir("checkpoints", 0777);

    for (int epoch=start_epoch; epoch<epochs; epoch++) {
        reset_flags();
        printf("\n========== Epoch %d/%d ==========\n", epoch+1, epochs);
        agent_set_schedule(agent, epoch);
        printf("Entropy coeff: %.6f | Temperature: %.6f\n", agent->entropy_coeff, agent->temperature);
        train_epoch(agent, optim);

        // On enregistre le modèle
        save_checkpoint("checkpoints/checkpoint.bin", agent, optim, epoch + 1);
        printf("Checkpoint saved to checkpoints/checkpoint.bin at the end of Epoch %d\n", epoch + 1);
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

    int start_epoch = 0;
    char* checkpoint_path = "checkpoints/checkpoint.bin";

    // On charge le modèle s'il existe
    FILE* check_fp = fopen(checkpoint_path, "rb");
    if (check_fp) {
        fclose(check_fp);
        if (load_checkpoint(checkpoint_path, agent, optim, &start_epoch) == 0) {
            printf("Loaded checkpoint at Epoch %d...\n", start_epoch + 1);
        } else {
            printf("Failed to load checkpoint.\n");
            start_epoch = 0;
        }
    } else {
        printf("No existing checkpoint found.\n");
    }

    train(agent, optim, start_epoch, EPOCHS);

    free(optim);
    free(agent);
    return 0;
}