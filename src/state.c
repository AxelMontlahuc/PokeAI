#include "state.h"

pokemon fetchPoke(MGBAConnection conn, int i) {
    pokemon poke;
    poke.maxHP = get_max_HP(conn.sock, i);
    poke.HP = get_HP(conn.sock, i);
    poke.level = get_level(conn.sock, i);
    poke.ATK = get_ATK(conn.sock, i);
    poke.DEF = get_DEF(conn.sock, i);
    poke.SPEED = get_SPEED(conn.sock, i);
    poke.ATK_SPE = get_ATK_SPE(conn.sock, i);
    poke.DEF_SPE = get_DEF_SPE(conn.sock, i);
    return poke;
}

state fetchState(MGBAConnection conn) {
    state s;
    /*s.map0 = mgba_read_bg0(&conn);
    s.map1 = mgba_read_map(&conn, 1);
    s.map2 = mgba_read_map(&conn, 2);
    s.map3 = mgba_read_map(&conn, 3);
    assert(s.map0 != NULL && s.map1 != NULL && s.map2 != NULL && s.map3 != NULL);

    s.bg0 = s.map0->tiles;
    s.bg1 = s.map1->tiles;
    s.bg2 = s.map2->tiles;
    s.bg3 = s.map3->tiles;*/

    s.team = malloc(6 * sizeof(pokemon));
    assert(s.team != NULL);
    for (int i=0; i<6; i++) {
        s.team[i] = fetchPoke(conn, i+1);
    }
    
    s.enemy = malloc(3 * sizeof(int));
    assert(s.enemy != NULL);
    s.enemy[0] = get_enemy_max_HP(conn.sock);
    s.enemy[1] = get_enemy_HP(conn.sock);
    s.enemy[2] = get_enemy_level(conn.sock);

    s.PP = malloc(4 * sizeof(int));
    for (int i=0; i<4; i++) {
        s.PP[i] = get_PP(conn.sock, i+1);
    }

    s.zone = get_zone(conn.sock);
    s.clock = get_clock(conn.sock);

    return s;
}

double* convertState(state s) {
    //double* out = malloc((4*(32*32) + (6*8) + 4 + 3 + 2) * sizeof(double));
    double* out = malloc((6*8 + 4 + 3 + 2) * sizeof(double));
    /*for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[0*32*32 + k*32 + l] = (double)s.bg0[k][l] / 2048.0 - 0.5;
        }
    }
    for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[1*32*32 + k*32 + l] = (double)s.bg1[k][l] / 2048.0 - 0.5;
        }
    }
    for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[2*32*32 + k*32 + l] = (double)s.bg2[k][l] / 2048.0 - 0.5;
        }
    }
    for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[3*32*32 + k*32 + l] = (double)s.bg3[k][l] / 2048.0 - 0.5;
        }
    }*/
    for (int i=0; i<6; i++) {
        out[i*8 + 0] = (double)s.team[i].maxHP / 300.0;
        out[i*8 + 1] = (double)s.team[i].HP / 300.0;
        out[i*8 + 2] = (double)s.team[i].level / 100.0;
        out[i*8 + 3] = (double)s.team[i].ATK / 300.0;
        out[i*8 + 4] = (double)s.team[i].DEF / 300.0;
        out[i*8 + 5] = (double)s.team[i].ATK_SPE / 300.0;
        out[i*8 + 6] = (double)s.team[i].DEF_SPE / 300.0;
        out[i*8 + 7] = (double)s.team[i].SPEED / 300.0;
    }
    out[6*8 + 0] = (double)s.PP[0] / 64.0;
    out[6*8 + 1] = (double)s.PP[1] / 64.0;
    out[6*8 + 2] = (double)s.PP[2] / 64.0;
    out[6*8 + 3] = (double)s.PP[3] / 64.0;
    out[6*8 + 4] = (double)s.enemy[0] / 300.0;
    out[6*8 + 5] = (double)s.enemy[1] / 300.0;
    out[6*8 + 6] = (double)s.enemy[2] / 100.0;
    out[6*8 + 7] = (double)s.zone / 255.0;
    out[6*8 + 8] = (double)s.clock / 255.0;

    return out;
}