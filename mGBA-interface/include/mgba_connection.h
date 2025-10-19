#ifndef MGBA_CONNECTION_H
#define MGBA_CONNECTION_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <windows.h>
#include <winsock2.h>

typedef struct {
    SOCKET sock;
    char ip_address[16];
    int port;
} MGBAConnection;


int mgba_connect(MGBAConnection* conn, const char* ip_address, int port);
int mgba_send_command(MGBAConnection* conn, const char* message, char* response, int response_size);
void mgba_disconnect(MGBAConnection* conn);

#endif