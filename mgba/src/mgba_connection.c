#include "../include/mgba_connection.h"

int mgba_connect(MGBAConnection* conn, const char* ip_address, int port) {
    #ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            return WSAGetLastError();
        }
    #endif

    if ((conn->sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        #ifdef _WIN32
            WSACleanup();
        #endif
        return WSAGetLastError();
    }

    struct sockaddr_in server;
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    #ifdef _WIN32
        server.sin_addr.s_addr = inet_addr(ip_address);
    #else
        if (inet_pton(AF_INET, ip_address, &server.sin_addr) != 1) {
            closesocket(conn->sock);
            return EINVAL;
        }
    #endif

    if (connect(conn->sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        int err = WSAGetLastError();
        closesocket(conn->sock);
        #ifdef _WIN32
            WSACleanup();
        #endif
        return err;
    }

    strncpy(conn->ip_address, ip_address, sizeof(conn->ip_address));
    conn->ip_address[sizeof(conn->ip_address) - 1] = '\0';
    conn->port = port;

    return 0;
}

int mgba_send_command(MGBAConnection* conn, const char* message, char* response, int response_size) {
    int recv_size;
    if (send(conn->sock, message, strlen(message), 0) < 0) {
        return -WSAGetLastError();
    }
    if ((recv_size = recv(conn->sock, response, response_size - 1, 0)) == SOCKET_ERROR) {
        return -WSAGetLastError();
    }
    response[recv_size] = '\0';
    return recv_size;
}

void mgba_disconnect(MGBAConnection* conn) {
    if (!conn) return;
    closesocket(conn->sock);
    #ifdef _WIN32
        WSACleanup();
    #endif
}