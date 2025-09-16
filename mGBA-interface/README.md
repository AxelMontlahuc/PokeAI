# mGBA Controller Library

A C library for programmatically controlling and reading data from mGBA GameBoy Advance emulator.

> WARNING: Linux and macOS are not supported due to the use of WinSock.

## Features

- Connect to mGBA via socket interface
- Send button press and hold commands
- Read and analyze the current game map
- Display map as ASCII characters
- Read the pokemon party HP's and levels

## Requirements

- Windows system (uses WinSock)
- mGBA emulator with the socket server script loaded
- C compiler (GCC recommended)
- CMake (optional, for easier building)

## Getting Started

1. Start mGBA and load your ROM
2. In mGBA, go to `Tools > Scripting` and load `mGBASocketServer.lua`
3. Build the library and examples using CMake or directly with GCC
4. Run one of the examples or integrate the library into your project

## Building

### Using CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Manual Building

```bash
gcc -c src/mgba_connection.c -o build/mgba_connection.o -Iinclude
gcc -c src/mgba_controller.c -o build/mgba_controller.o -Iinclude
gcc -c src/mgba_map.c -o build/mgba_map.o -Iinclude
gcc -c src/mgba_interactions.c -o build/mgba_interactions.o -Iinclude
gcc -c src/mgba_intel.c -o build/mgba_intel.o -Iinclude
ar rcs lib/libmgba_controller.a build/mgba_connection.o build/mgba_controller.o build/mgba_map.o
gcc examples/simple_controller.c -o examples/simple_controller -Iinclude -Llib -lmgba_controller -lws2_32
```

## Examples

The library includes two example programs:

- `simple_controller`: Demonstrates how to send button commands to move the character
- `map_viewer`: Shows how to read and display the current game map
- `intel_gatherer`: Shows how to read the pokemon party HP's and levels

## Usage

```c
#include "mgba_controller.h"
#include "mgba_connection.h"

int main() {
    MGBAConnection conn;
    mgba_connect(&conn, "127.0.0.1", 8888);
    
    // Press some buttons
    mgba_press_button(&conn, MGBA_BUTTON_RIGHT, 50);
    mgba_press_button(&conn, MGBA_BUTTON_A, 50);
    
    mgba_disconnect(&conn);
    return 0;
}
```

## License

This project is available under the MIT License.