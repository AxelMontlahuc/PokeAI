#include "../include/mgba_intel.h"

int parse_next_int(const char** p) {
    const char* s = *p;
    int sign = 1;
    int v = 0;
    if (*s == '-') { 
        sign = -1; 
        s++; 
    }

    while (*s && *s != ',') {
        if (*s >= '0' && *s <= '9') v = v * 10 + (*s - '0');
        s++;
    }

    if (*s == ',') s++;
    *p = s;
    return sign * v;
}

int read_state(SOCKET sock, int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int bg0_out[32][32], int bg2_out[32][32]) {
    const char* cmd = "bulk.readState";

    if (send(sock, cmd, (int)strlen(cmd), 0) < 0) {
        return -WSAGetLastError();
    }

    const int BUF = 65536;
    char* buf = (char*)malloc(BUF);
    
    int recv_size = recv(sock, buf, BUF - 1, 0);
    if (recv_size == SOCKET_ERROR) {
        int err = -WSAGetLastError();
        free(buf);
        return err;
    }
    buf[recv_size] = '\0';

    const char* p = buf;

    for (int i = 0; i < 6 * 8; i++) team_out[i] = parse_next_int(&p);
    for (int i = 0; i < 3; i++) enemy_out[i] = parse_next_int(&p);
    for (int i = 0; i < 4; i++) pp_out[i] = parse_next_int(&p);

    *zone_out = parse_next_int(&p);
    *clock_out = parse_next_int(&p);

    for (int k = 0; k < 32; k++) {
        for (int l = 0; l < 32; l++) {
            bg0_out[k][l] = parse_next_int(&p);
        }
    }
    for (int k = 0; k < 32; k++) {
        for (int l = 0; l < 32; l++) {
            bg2_out[k][l] = parse_next_int(&p);
        }
    }

    free(buf);
    return 0;
}