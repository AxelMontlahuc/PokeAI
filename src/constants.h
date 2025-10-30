#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

#define ACTION_COUNT 6
extern const int INPUT_SIZE;
extern const int HIDDEN_SIZE;

extern int ID;

extern const char* const QUEUE_DIR;
extern const char* const LOCKS_DIR;
extern const char* const CHECKPOINT_PATH;

extern const char* ROM_PATH;
extern const char* CORE_PATH;
extern const char* SCREEN_PATH_PREFIX;
extern char SCREEN_PATH[512];
extern const char* SAVESTATE_PATH;

extern const int SPEED;

extern int WORKER_TRAJECTORIES;
extern int WORKER_STEPS;
extern int WORKER_BATCH_SIZE;

extern double TEMP_MAX;
extern double TEMP_MIN;
extern double TEMP_DECAY;

extern int FILES_PER_STEP;

extern int PPO_EPOCHS;
extern double BASE_LR;
extern double LR_DECAY;
extern int WARMUP_EPISODES;
extern double MIN_WARMUP_FACTOR;
extern int MB_TRAJ_THRESHOLD;
extern int MB_SIZE_DEFAULT;
extern double CLIP_EPS;
extern double TARGET_KL;

extern double GAMMA_DISCOUNT;
extern double GAE_LAMBDA;

extern double ENTROPY_COEFF;
extern double ENTROPY_DECAY;
extern double ENTROPY_MIN;
extern double VALUE_COEFF;
extern double VALUE_CLIP_EPS;
extern double GRAD_CLIP_NORM;

extern double ADAM_BETA1;
extern double ADAM_BETA2;
extern double ADAM_EPS;

extern double NUM_EPS;
extern double STD_EPS;

extern double BASE_LR_MIN;
extern double BASE_LR_MAX;

extern const uint32_t TRAJ_MAGIC;
extern const uint16_t TRAJ_VERSION;
extern const char CKPT_MAGIC[8];
extern const uint32_t CKPT_VERSION;

#endif
