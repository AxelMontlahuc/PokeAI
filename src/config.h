#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_SIZE 387
#define HIDDEN_SIZE 128
#define POLICY_OUTPUT_SIZE 6
#define VALUE_OUTPUT_SIZE 1
#define MAX_OUTPUT_SIZE 6
#define COL_SIZE (INPUT_SIZE + HIDDEN_SIZE)  // INPUT_SIZE + HIDDEN_SIZE

#define EPOCHS 500
#define NUM_ENVS 8
#define PPO_EPOCHS 3
#define TRAJ_SIZE 512
#define BATCH_SIZE (TRAJ_SIZE * NUM_ENVS)
#define MINIBATCH_SIZE 256
#define SEQ_LEN 32
#define NUM_SEQS (MINIBATCH_SIZE / SEQ_LEN)

#define GAMMA 0.995f
#define LAMBDA 0.95f // pour la GAE
#define EPSILON 0.2f

#define LEARNING_RATE 0.0003f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON_ADAM 1e-8f

#define ENTROPY_COEFF 0.04f

// Permet de stopper prématurément les epochs de PPO si le KL devient trop grand
#define KL_TARGET 0.03f
#define KL_MIN_EPOCHS 1
#define MAX_GRAD_NORM 0.5f

// Entropie dynamique
#define ENTROPY_INIT 0.04f
#define ENTROPY_MIN 0.005f
#define ENTROPY_DECAY_EPOCHS 500

// Système de température
#define TEMP_INIT 1.0f
#define TEMP_MIN 1.0f
#define TEMP_DECAY_EPOCHS 0

// Récompenses
#define WEIGHT_EXPLORATION 0.0f    // Récompense par nouvelle tile
#define WEIGHT_LEVEL_UP    2.5f     // Récompense par niveau gagné
#define STEP_PENALTY       -0.0015f // Pénalité par pas (pour pousser l'agent à être plus rapide)

#endif