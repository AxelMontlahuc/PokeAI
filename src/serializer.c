#include "serializer.h"

int write_batch_file(const char* path_tmp, const char* path_final, trajectory** batch, int batch_size, int steps, double temperature) {
    FILE* f = fopen(path_tmp, "wb");
    if (!f) return -1;

    uint32_t magic = TRAJ_MAGIC;
    uint16_t version = TRAJ_VERSION;
    uint16_t reserved = 0u;
    uint32_t steps_u32 = (uint32_t)steps;
    uint32_t batch_u32 = (uint32_t)batch_size;
    uint32_t action_u32 = (uint32_t)ACTION_COUNT;
    uint32_t state_u32 = (uint32_t)sizeof(state);

    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&reserved, sizeof(reserved), 1, f);
    fwrite(&steps_u32, sizeof(steps_u32), 1, f);
    fwrite(&batch_u32, sizeof(batch_u32), 1, f);
    fwrite(&action_u32, sizeof(action_u32), 1, f);
    fwrite(&state_u32, sizeof(state_u32), 1, f);
    fwrite(&temperature, sizeof(double), 1, f);

    for (int b = 0; b < batch_size; b++) {
        trajectory* t = batch[b];
        for (int i = 0; i < steps; i++) {
            uint16_t action = (uint16_t)t->actions[i];
            fwrite(&t->states[i], sizeof(state), 1, f);
            fwrite(t->probs[i], sizeof(double), ACTION_COUNT, f);
            fwrite(&t->values[i], sizeof(double), 1, f);
            fwrite(&action, sizeof(uint16_t), 1, f);
            fwrite(&t->rewards[i], sizeof(double), 1, f);
        }
    }

    fclose(f);

    if (rename(path_tmp, path_final) != 0) {
        remove(path_tmp);
        return -1;
    }

    return 0;
}

int read_batch_file(const char* path, trajectory*** out_batch, int* out_batch_size, int* out_steps, double* out_temperature) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic; 
    uint16_t version; 
    uint16_t reserved;
    uint32_t steps_u32;
    uint32_t batch_u32;
    uint32_t action_u32;
    uint32_t state_u32; 
    double temperature;

    fread(&magic, sizeof(magic), 1, f);
    fread(&version, sizeof(version), 1, f);
    fread(&reserved, sizeof(reserved), 1, f);
    fread(&steps_u32, sizeof(steps_u32), 1, f);
    fread(&batch_u32, sizeof(batch_u32), 1, f);
    fread(&action_u32, sizeof(action_u32), 1, f);
    fread(&state_u32, sizeof(state_u32), 1, f);
    fread(&temperature, sizeof(double), 1, f);

    int steps = (int)steps_u32;
    int batch_size = (int)batch_u32;

    trajectory** batch = malloc(batch_size * sizeof(trajectory*));
    assert(batch != NULL);

    for (int b = 0; b < batch_size; b++) {
        trajectory* t = initTrajectory(steps);
        for (int i = 0; i < steps; i++) {
            t->probs[i] = malloc(ACTION_COUNT * sizeof(double));
            assert(t->probs[i] != NULL);

            uint16_t action;
            fread(&t->states[i], sizeof(state), 1, f);
            fread(t->probs[i], sizeof(double), ACTION_COUNT, f);
            fread(&t->values[i], sizeof(double), 1, f);
            fread(&action, sizeof(uint16_t), 1, f);
            fread(&t->rewards[i], sizeof(double), 1, f);
            t->actions[i] = (int)action;
        }
        batch[b] = t;
    }

    fclose(f);

    *out_batch = batch;
    *out_batch_size = batch_size;
    *out_steps = steps;
    *out_temperature = temperature;
    
    return 0;
}