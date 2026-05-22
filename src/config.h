#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_SIZE 387
#define HIDDEN_SIZE 128
#define POLICY_OUTPUT_SIZE 6
#define VALUE_OUTPUT_SIZE 1
#define MAX_OUTPUT_SIZE 6
#define COL_SIZE (INPUT_SIZE + HIDDEN_SIZE)  // INPUT_SIZE + HIDDEN_SIZE

#define EPOCHS 500
#define NUM_ENVS 16
#define PPO_EPOCHS 6
#define TRAJ_SIZE 2048
#define BATCH_SIZE (TRAJ_SIZE * NUM_ENVS)
#define MINIBATCH_SIZE 256
#define SEQ_LEN 32
#define NUM_SEQS (MINIBATCH_SIZE / SEQ_LEN)

#define GAMMA 0.995
#define LAMBDA 0.95 // pour la GAE
#define EPSILON 0.2

#define LEARNING_RATE 0.0003
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON_ADAM 1e-8

#define ENTROPY_COEFF 0.04

// Permet de stopper prématurément les epochs de PPO si le KL devient trop grand
#define KL_TARGET 0.03
#define KL_MIN_EPOCHS 1
#define MAX_GRAD_NORM 0.5

// Entropie dynamique
#define ENTROPY_INIT 0.04
#define ENTROPY_MIN 0.005
#define ENTROPY_DECAY_EPOCHS 500

// Système de température
#define TEMP_INIT 1.0
#define TEMP_MIN 1.0
#define TEMP_DECAY_EPOCHS 20

// Récompenses
#define WEIGHT_EXPLORATION 0.0    // Récompense par nouvelle tile
#define WEIGHT_LEVEL_UP    2.5     // Récompense par niveau gagné
#define STEP_PENALTY       -0.0015 // Pénalité par pas (pour pousser l'agent à être plus rapide)

#endif