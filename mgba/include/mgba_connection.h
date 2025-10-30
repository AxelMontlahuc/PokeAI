#ifndef MGBA_CONNECTION_H
#define MGBA_CONNECTION_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
    #include <winsock2.h>
    #include <windows.h>
    #include <ws2tcpip.h>
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
    #include <errno.h>
    typedef int SOCKET;
    #define INVALID_SOCKET (-1)
    #define SOCKET_ERROR   (-1)
    #define closesocket close
    #define WSAGetLastError() (errno)
#endif

typedef struct {
    SOCKET sock;
    char ip_address[16];
    int port;
} MGBAConnection;

int mgba_connect(MGBAConnection* conn, const char* ip_address, int port);
int mgba_send_command(MGBAConnection* conn, const char* message, char* response, int response_size);
void mgba_disconnect(MGBAConnection* conn);

#endif