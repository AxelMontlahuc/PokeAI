#include "constants.h"

const int INPUT_SIZE = 6*8 + 4 + 3 + 2 + 2*32*32;
const int HIDDEN_SIZE = 256;

int ID = 1;
int PORT = 8888;
const char* HOST_ADDR = "127.0.0.1";

const char* QUEUE_DIR = "queue";
const char* LOCKS_DIR = "locks";
char* CHECKPOINT_PATH = "checkpoints/model-last.sav";

const char* ROM_PATH = "ROM/pokemon.gba";
const char* CORE_PATH = "../libretro-super/dist/unix/mgba_libretro.so";
const char* SCREEN_PATH_PREFIX = "screen/";
char SCREEN_PATH[512] = "screen/1.bmp";
const char* SAVESTATE_PATH = "ROM/start.sav";

const int SPEED = 120;
const int BUTTON_PRESS_MS = 70;

int WORKER_TRAJECTORIES = 8;
int WORKER_STEPS = 256;
int WORKER_BATCH_SIZE = 4;

double TEMP_MAX = 1.5;
double TEMP_MIN = 1.0;
double TEMP_DECAY = 0.95;

int FILES_PER_STEP = 4;

int PPO_EPOCHS = 4;
double BASE_LR = 0.001;
double LR_DECAY = 0.9995;
int WARMUP_EPISODES = 5;
double MIN_WARMUP_FACTOR = 0.1;
int MB_TRAJ_THRESHOLD = 8;
int MB_SIZE_DEFAULT = 8;
double CLIP_EPS = 0.20;
double TARGET_KL = 0.03;

double GAMMA_DISCOUNT = 0.995;
double GAE_LAMBDA = 0.95;

double ENTROPY_COEFF = 0.02;
double ENTROPY_DECAY = 0.995;
double ENTROPY_MIN = 0.005;
double VALUE_COEFF = 0.50;
double VALUE_CLIP_EPS = 0.20;
double GRAD_CLIP_NORM = 100.0;

double ADAM_BETA1 = 0.9;
double ADAM_BETA2 = 0.999;
double ADAM_EPS = 1e-8;

double NUM_EPS = 1e-12;
double STD_EPS = 1e-8;

double BASE_LR_MIN = 3e-4;
double BASE_LR_MAX = 1e-2;

const uint32_t TRAJ_MAGIC = 0x30524A54u; // 'TRJ0'
const uint16_t TRAJ_VERSION = 4;
const char CKPT_MAGIC[8] = { 'L','S','T','M','S','A','V','\0' };
const uint32_t CKPT_VERSION = 2u;