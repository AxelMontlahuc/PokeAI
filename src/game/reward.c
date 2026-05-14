#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "config.h"
#include "reward.h"

double reward(double old_state[INPUT_SIZE], double new_state[INPUT_SIZE]) {
    // 387 - FLAG maison du joueur
	// 388 - FLAG chambre du joueur
	// 389 - FLAG horloge
	// 390 - FLAG Littleroot (village de départ)
	// 391 - FLAG maison du rival
	// 392 - FLAG route 101
	// 393 - FLAG laboratoire
	// 394 - FLAG Oldale (premier village)
	// 395 - FLAG route 103
	// 396 - FLAG deuxième fois au laboratoire (après avoir battu le rival)
	// 397 - FLAG route 102
	// 398 - FLAG Petalburg (deuxième ville)
	// 399 - FLAG route 104
	// 400 - FLAG Petalburg Woods
	// 401 - FLAG Rustboro (troisième ville)

    return 0;
}

bool is_done(double state[INPUT_SIZE]) {
    return false;
}