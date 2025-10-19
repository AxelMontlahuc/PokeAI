# mGBA Controller Library

A C library for programmatically controlling and reading data from mGBA GameBoy Advance emulator.

> WARNING: Linux and macOS are not supported due to the use of WinSock.

## Features

- Connect to mGBA via socket interface
- Send button press and hold commands
- Read the current game map
- Read the pokemon party HP's and levels
- Read other informations at precise memory addresses

## Requirements

- Windows system (uses WinSock)
- mGBA emulator with the socket server script loaded
- C compiler (GCC recommended)

## Getting Started

1. Start mGBA and load your ROM
2. In mGBA, go to `Tools > Scripting` and load `mGBASocketServer.lua`
3. Build the library with GCC

## Basic Usage

```c
#include "mgba_controller.h"
#include "mgba_connection.h"

int main() {
    MGBAConnection conn;
    mgba_connect(&conn, "127.0.0.1", 8888);
    
    mgba_press_button(&conn, MGBA_BUTTON_RIGHT, 50);
    mgba_press_button(&conn, MGBA_BUTTON_A, 50);
    
    mgba_disconnect(&conn);
    return 0;
}
```

## License

This project is available under the MIT License.