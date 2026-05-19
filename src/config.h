#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_SIZE 387
#define HIDDEN_SIZE 128
#define POLICY_OUTPUT_SIZE 6
#define VALUE_OUTPUT_SIZE 1
#define MAX_OUTPUT_SIZE 6
#define COL_SIZE (INPUT_SIZE + HIDDEN_SIZE)  // INPUT_SIZE + HIDDEN_SIZE

#define EPOCHS 50
#define NUM_ENVS 8
#define PPO_EPOCHS 8
#define TRAJ_SIZE 2048
#define BATCH_SIZE (TRAJ_SIZE * NUM_ENVS)
#define MINIBATCH_SIZE 512

#define GAMMA 0.99
#define LAMBDA 0.95 // pour la GAE
#define EPSILON 0.2

#define LEARNING_RATE 0.0003
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON_ADAM 1e-8

#define ENTROPY_COEFF 0.02

// Permet de stopper prématurément les epochs de PPO si le KL devient trop grand
#define KL_TARGET 0.02
#define KL_MIN_EPOCHS 1

// Entropi dynamique
#define ENTROPY_INIT 0.02
#define ENTROPY_MIN 0.002
#define ENTROPY_DECAY_EPOCHS 100

// Système de température
#define TEMP_INIT 2.0
#define TEMP_MIN 1.0
#define TEMP_DECAY_EPOCHS 20

#endif