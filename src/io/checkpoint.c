#include <stdio.h>
#include <stdlib.h>
#include "checkpoint.h"

int save_checkpoint(char* filepath, Agent* agent, Optimizer* optim, int epoch) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        perror("Erreur : Impossible d'ouvrir le fichier de sauvegarde");
        return -1;
    }

    CheckpointHeader header;
    header.magic = 0x504F4B45; // 'POKE'
    header.epoch = epoch;

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (fwrite(agent, sizeof(Agent), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (fwrite(optim, sizeof(Optimizer), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int load_checkpoint(char* filepath, Agent* agent, Optimizer* optim, int* epoch) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        return -1;
    }

    CheckpointHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1 || header.magic != 0x504F4B45) {
        fprintf(stderr, "Erreur : Fichier de checkpoint invalide ou corrompu !\n");
        fclose(fp);
        return -1;
    }

    if (fread(agent, sizeof(Agent), 1, fp) != 1) {
        fprintf(stderr, "Erreur : Impossible de lire les données de l'Agent !\n");
        fclose(fp);
        return -1;
    }

    if (fread(optim, sizeof(Optimizer), 1, fp) != 1) {
        fprintf(stderr, "Erreur : Impossible de lire les données de l'Optimiseur !\n");
        fclose(fp);
        return -1;
    }

    *epoch = header.epoch;
    fclose(fp);
    return 0;
}
