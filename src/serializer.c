#include "serializer.h"

static int write_exact(FILE* f, const void* buf, size_t n) {
    return fwrite(buf, 1, n, f) == n ? 0 : -1;
}
static int read_exact(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

int write_batch_file(const char* path_tmp, const char* path_final, trajectory** batch, int batch_size, int steps, double epsilon, double temperature) {
    FILE* f = fopen(path_tmp, "wb");
    assert(f != NULL);

    TrajFileHeader hdr = {0};
    hdr.magic = 0x30524A54u; // 'TRJ0' in little-endian
    hdr.version = 2;
    hdr.steps = (uint32_t)steps;
    hdr.batch_size = (uint32_t)batch_size;
    hdr.action_count = ACTION_COUNT;
    hdr.state_size = (uint32_t)sizeof(state);
    hdr.epsilon = epsilon;
    hdr.temperature = temperature;

    if (write_exact(f, &hdr, sizeof(hdr)) != 0) {
        fclose(f); 
        return -1;
    }

    for (int b = 0; b < batch_size; b++) {
        trajectory* t = batch[b];
        
        for (int i = 0; i < steps; i++) {
            uint16_t a = (uint16_t)t->actions[i];
            if (write_exact(f, &t->states[i], sizeof(state)) != 0 ||
                write_exact(f, t->probs[i], sizeof(double) * ACTION_COUNT) != 0 ||
                write_exact(f, t->behav_probs[i], sizeof(double) * ACTION_COUNT) != 0 ||
                write_exact(f, &a, sizeof(uint16_t)) != 0 ||
                write_exact(f, &t->rewards[i], sizeof(double)) != 0) {
                fclose(f);
                return -1;
            }
        }
    }

    if (fclose(f) != 0) return -1;

    if (rename(path_tmp, path_final) != 0) {
        remove(path_tmp);
        return -1;
    }
    return 0;
}

int read_batch_file(const char* path, trajectory*** out_batch, int* out_batch_size, int* out_steps, double* out_epsilon, double* out_temperature) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    TrajFileHeader hdr;
        if (read_exact(f, &hdr, sizeof(hdr)) != 0) { 
            fclose(f); 
            return -1; 
        }

        if (hdr.magic != 0x30524A54u || hdr.version != 2 || hdr.action_count != ACTION_COUNT || hdr.state_size != sizeof(state)) {
            fclose(f);
            return -1;
        }

    int steps = (int)hdr.steps;
    int batch_size = (int)hdr.batch_size;

    trajectory** batch = (trajectory**)malloc(sizeof(trajectory*) * batch_size);
    if (!batch) { 
        fclose(f); 
        return -1; 
    }

    for (int b = 0; b < batch_size; b++) {
        trajectory* t = initTrajectory(steps);
        if (!t) { 
            fclose(f); 
            return -1; 
        }

        for (int i = 0; i < steps; i++) {
            t->probs[i] = (double*)malloc(sizeof(double) * ACTION_COUNT);
            t->behav_probs[i] = (double*)malloc(sizeof(double) * ACTION_COUNT);
            uint16_t a16;

            if (read_exact(f, &t->states[i], sizeof(state)) != 0 || !t->probs[i] || !t->behav_probs[i] ||
                    read_exact(f, t->probs[i], sizeof(double) * ACTION_COUNT) != 0 ||
                    read_exact(f, t->behav_probs[i], sizeof(double) * ACTION_COUNT) != 0 ||
                    read_exact(f, &a16, sizeof(uint16_t)) != 0 ||
                    read_exact(f, &t->rewards[i], sizeof(double)) != 0) { 
                fclose(f); 
                return -1; 
            }

            t->actions[i] = (MGBAButton)a16;
        }

        batch[b] = t;
    }

    fclose(f);

    *out_batch = batch;
    *out_batch_size = batch_size;
    *out_steps = steps;

    if (out_epsilon) *out_epsilon = hdr.epsilon;
    if (out_temperature) *out_temperature = hdr.temperature;
    
    return 0;
}