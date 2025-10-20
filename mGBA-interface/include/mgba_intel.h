#ifndef MGBA_INTEL_H
#define MGBA_INTEL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mgba_connection.h"

int get_max_HP(SOCKET sock, int pokemon);
int get_HP(SOCKET sock, int pokemon);
int get_level(SOCKET sock, int pokemon);
int get_ATK(SOCKET sock, int pokemon);
int get_DEF(SOCKET sock, int pokemon);
int get_SPEED(SOCKET sock, int pokemon);
int get_ATK_SPE(SOCKET sock, int pokemon);
int get_DEF_SPE(SOCKET sock, int pokemon);
int get_PP(SOCKET sock, int move);
int get_enemy_max_HP(SOCKET sock);
int get_enemy_HP(SOCKET sock);
int get_enemy_level(SOCKET sock);
int get_zone(SOCKET sock);
int get_clock(SOCKET sock);

#endif