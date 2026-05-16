#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_SIZE 387
#define HIDDEN_SIZE 64
#define POLICY_OUTPUT_SIZE 6
#define VALUE_OUTPUT_SIZE 1
#define MAX_OUTPUT_SIZE 6
#define COL_SIZE 461  // INPUT_SIZE + HIDDEN_SIZE

#define EPOCHS 50
#define NUM_ENVS 8
#define PPO_EPOCHS 2
#define TRAJ_SIZE 256
#define BATCH_SIZE (TRAJ_SIZE * NUM_ENVS)
#define MINIBATCH_SIZE 128

#define GAMMA 0.99
#define LAMBDA 0.95 // pour GAE
#define EPSILON 0.3

#define LEARNING_RATE 0.0003
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON_ADAM 1e-8

#define ENTROPY_COEFF 0.01

#endif