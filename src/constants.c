#include "constants.h"

const int INPUT_SIZE = 6*8 + 4 + 3 + 2 + 2*32*32;
const int HIDDEN_SIZE = 128;

const char* const HOST_ADDR = "127.0.0.1";
int PORT = 8888;

const char* const QUEUE_DIR = "queue";
const char* const LOCKS_DIR = "locks";
const char* const CHECKPOINT_PATH = "checkpoints/model-last.sav";

int WORKER_TRAJECTORIES = 64;
int WORKER_STEPS = 256;
int WORKER_BATCH_SIZE = 8;
int BUTTON_PRESS_MS = 50;

double TEMP_MAX = 3.0;
double TEMP_MIN = 1.0;
double TEMP_DECAY = 0.97;

int FILES_PER_STEP = 4;

int PPO_EPOCHS = 3;
double BASE_LR = 0.010;
double LR_DECAY = 0.99;
int WARMUP_EPISODES = 5;
double MIN_WARMUP_FACTOR = 0.1;
int MB_TRAJ_THRESHOLD = 8;
int MB_SIZE_DEFAULT = 4;
double CLIP_EPS = 0.20;
double TARGET_KL = 0.02;

double GAMMA_DISCOUNT = 0.90;
double GAE_LAMBDA = 0.95;

double ENTROPY_COEFF = 0.05;
double VALUE_COEFF = 0.50;
double VALUE_CLIP_EPS = 0.20;
double GRAD_CLIP_NORM = 1.0;

double ADAM_BETA1 = 0.9;
double ADAM_BETA2 = 0.999;
double ADAM_EPS = 1e-8;

double NUM_EPS = 1e-12;
double STD_EPS = 1e-8;

const uint32_t TRAJ_MAGIC = 0x30524A54u; // 'TRJ0'
const uint16_t TRAJ_VERSION = 4;
const char CKPT_MAGIC[8] = { 'L','S','T','M','S','A','V','\0' };
const uint32_t CKPT_VERSION = 2u;