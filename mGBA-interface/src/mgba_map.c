#include "../include/mgba_map.h"

static char* latin1_to_hex(const char* input, int length) {
    char* output = malloc((length * 2 + 1) * sizeof(char));
    if (output == NULL) return NULL;
    
    for (int i = 0; i < length; i++) {
        sprintf(output + (i * 2), "%02x", (unsigned char)input[i]);
    }
    output[length * 2] = '\0';
    return output;
}

int mgba_get_screen_offset(MGBAConnection* conn, int* x_offset, int* y_offset) {
    char response[32];
    
    if (mgba_send_command(conn, "memoryDomain.read8,io,0x4000018", response, sizeof(response)) < 0) {
        return -1;
    }
    *x_offset = atoi(response);
    
    if (mgba_send_command(conn, "memoryDomain.read8,io,0x400001A", response, sizeof(response)) < 0) {
        return -1;
    }
    *y_offset = atoi(response);
    
    return 0;
}

char* request_range(SOCKET sock, char* address, char* length) {
    char* message = malloc(128 * sizeof(char));
    sprintf(message, "memoryDomain.readRange,vram,%s,%s", address, length);

    char* server_reply = malloc((32*32*16) * sizeof(char));
    int recv_size;
    if (send(sock, message, strlen(message), 0) < 0) {
        printf("Send failed. Error Code: %d\n", WSAGetLastError());
    }
    if ((recv_size = recv(sock, server_reply, 8192 - 1, 0)) == SOCKET_ERROR) {
        printf("Receive failed. Error Code: %d\n", WSAGetLastError());
    }
    server_reply[recv_size] = '\0';

    free(message);

    return server_reply;
}

MGBAMap* mgba_read_bg0(MGBAConnection* conn) {
    char response[8192];
    char command[64];
    char address[] = "0x0600e800";

    MGBAMap* map = (MGBAMap*)malloc(sizeof(MGBAMap));
    if (map == NULL) return NULL;
    
    map->width = 32;
    map->height = 32;
    
    map->tiles = (int**)malloc(map->height * sizeof(int*));
    if (map->tiles == NULL) {
        free(map);
        return NULL;
    }
    
    for (int i = 0; i < map->height; i++) {
        map->tiles[i] = (int*)malloc(map->width * sizeof(int));
        if (map->tiles[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(map->tiles[j]);
            }
            free(map->tiles);
            free(map);
            return NULL;
        }
    }
    
    snprintf(command, sizeof(command), "memoryDomain.readRange,vram,%s,2048", address);
    if (mgba_send_command(conn, command, response, sizeof(response)) < 0) {
        mgba_free_map(map);
        return NULL;
    }
    
    char* hex_response = latin1_to_hex(response, 2048);
    if (hex_response == NULL) {
        mgba_free_map(map);
        return NULL;
    }

    for (int row = 0; row < map->height; row++) {
        for (int col = 0; col < map->width; col++) {
            int src_row = (row) % map->height;
            int src_col = (col) % map->width;
            int tile_index = src_row * map->width + src_col;
            
            char buffer[5];
            strncpy(buffer, hex_response + (tile_index * 4), 4);
            buffer[4] = '\0';
            
            map->tiles[row][col] = (int)strtol(buffer, NULL, 16);
        }
    }
    
    free(hex_response);
    return map;
}

MGBAMap* mgba_read_map(MGBAConnection* conn, int bg) {
    int x_offset, y_offset;
    char response[8192];
    char command[64];

    if (mgba_get_screen_offset(conn, &x_offset, &y_offset) != 0) {
        return NULL;
    }

    int tile_x_offset = x_offset / 8;
    int tile_y_offset = y_offset / 8;

    MGBAMap* map = (MGBAMap*)malloc(sizeof(MGBAMap));
    if (map == NULL) return NULL;
    
    map->width = 32;
    map->height = 32;

    map->tiles = (int**)malloc(map->height * sizeof(int*));
    if (map->tiles == NULL) {
        free(map);
        return NULL;
    }
    
    for (int i = 0; i < map->height; i++) {
        map->tiles[i] = (int*)malloc(map->width * sizeof(int));
        if (map->tiles[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(map->tiles[j]);
            }
            free(map->tiles);
            free(map);
            return NULL;
        }
    }

    char* address;
    switch (bg) {
        case 1:
            address = "0x0600e800";
            break;
        case 2:
            address = "0x0600e000";
            break;
        case 3:
            address = "0x0600f000";
            break;
        default:
            address = "0x0600e000";
            break;
    }

    snprintf(command, sizeof(command), "memoryDomain.readRange,vram,%s,2048", address);
    if (mgba_send_command(conn, command, response, sizeof(response)) < 0) {
        mgba_free_map(map);
        return NULL;
    }

    char* hex_response = latin1_to_hex(response, 2048);
    if (hex_response == NULL) {
        mgba_free_map(map);
        return NULL;
    }

    for (int row = 0; row < map->height; row++) {
        for (int col = 0; col < map->width; col++) {
            int src_row = (row + tile_y_offset) % map->height;
            int src_col = (col + tile_x_offset) % map->width;
            int tile_index = src_row * map->width + src_col;
            
            char buffer[5];
            strncpy(buffer, hex_response + (tile_index * 4), 4);
            buffer[4] = '\0';
            
            map->tiles[row][col] = (int)strtol(buffer, NULL, 16);
        }
    }
    
    free(hex_response);
    return map;
}

void mgba_free_map(MGBAMap* map) {
    if (map == NULL) return;
    
    if (map->tiles != NULL) {
        for (int i = 0; i < map->height; i++) {
            if (map->tiles[i] != NULL) {
                free(map->tiles[i]);
            }
        }
        free(map->tiles);
    }
    
    free(map);
}