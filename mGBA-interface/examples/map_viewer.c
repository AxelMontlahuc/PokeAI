#include "../include/mgba_map.h"
#include "../include/mgba_connection.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

int main() {
    MGBAConnection conn;
    int result;
    
    // Connect to mGBA
    result = mgba_connect(&conn, "127.0.0.1", 8888);
    if (result != 0) {
        printf("Failed to connect to mGBA. Error code: %d\n", result);
        return 1;
    }
    
    printf("Connected to mGBA. Press Ctrl+C to exit.\n");
    
    // Continuously display map
    while (1) {
        system("@cls||clear");
        
        // Print map to console
        if (mgba_print_map(&conn, 2) != 0) {
            printf("Error printing map\n");
        }
        
        // Wait before refreshing
        Sleep(200);
    }
    
    // Clean up
    mgba_disconnect(&conn);
    return 0;
}