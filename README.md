# Projet
Le but de ce TIPE est de créer un réseau de neurones capable de jouer à Pokémon Emeraude en C sans utiliser de librairies de Machine Learning, de manière à tout recoder de zéro pour mieux comprendre les maths derrière l'agent que nous allons construire. 

Pour cela nous utilisons mGBA comme émulateur et grâce à la librairie winsock/socket (seules librairies non-standard utilisées) nous communiquons avec un script Lua dans mGBA pour récupérer l'état du jeu et envoyer des commandes.

Quand à l'agent, c'est un VPG (Vanilla Policy Gradient) utilisant pour politique un LSTM (Long Short-Term Memory), mettant en évidence les cycles et boucles.

# Utilisation
1. Cloner le dépôt :
```sh
~$ git clone https://github.com/AxelMontlahuc/PokeAI.git
```
2. Compiler le projet avec : 
```sh
~$ make
```
3. Placer une ROM de Pokémon Emeraude (version FR) nommée `pokemon.gba` dans le dossier `ROM`.
4. Lancer mGBA avec le script Lua `mGBASocketServer.lua` chargé.
5. Lancer l'agent avec : 
```sh
~$ ./bin/agent     # On Linux
~$ ./bin/agent.exe # On Windows
```

# Structure
- `mGBASocketServer.lua` : Script Lua utilisant l'API de mGBA pour faire le pont avec `mGBA-interface` à travers un socket.
- `mGBA-interface` : Librairie en C pour communiquer avec mGBA : appuyer sur des touches et lire la map et autre données du jeu. Pour plus d'informations, voir le [README](mGBA-interface/README.md).
- `src` : Code source de l'agent en C :
  - `agent` : Boucle principale de l'agent, gestion des épisodes, interaction avec l'environnement.
  - `policy` : Implémentation du LSTM et du VPG. 
  - `struct` : Définitions, initialisation et libération des structures de données.
  - `state` : Fonctions liées à l'état du jeu. 
  - `checkpoint.c` : Sauvegarde et restauration du modèle complet. 
  - `func` : Fonctions auxiliaires. 

# Documentation
Nous allons ici rentrer un peu plus en détail sur l'implémentation et les mathématiques derrière l'agent en détaillant chaque caractéristique du modèle. \
Voici le plan des caractéristiques que nous allons détailler :

1. Vanilla Policy Gradient
2. LSTM
3. Softmax Temperature
4. Epsilon-greedy
5. Entropy Bonus
6. Normalization
7. Adam

## Vanilla Policy Gradient
Voici le pseudo-code du VPG (aussi connu sous le nom REINFORCE) un petit peu modifié selon nos besoins : 
> _Entrée_ : Vecteur $\theta_0$ représentant les paramètres initiaux de la politique notée $\pi_\theta$.
> 1. pour chaque trajectoire $\tau_i = (s_0, a_0, \ldots, s_T, a_T)$ selon $\pi_\theta$ faire
> 2. $\quad$ Calculer la récompense de la trajectoire (rééquilibrée) : 
> $$G(\tau) = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$$
> 3. $\quad$ Calculer le gradient de la politique par méthode de Monte-Carlo : 
> $$\nabla_\theta\mathbb E_{\pi_\theta} G(\tau) = \frac 1 L \sum_\tau\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G(\tau)$$
> $\quad\quad\quad\space\space$ avec $L$ le nombre de trajectoires utilisées lors du calcul \
> $\quad\space\space$ 4. $\quad$ Mettre à jour $\theta$ par montée du gradient avec un taux d'apprentissage $\eta$ : 
> $$\theta \leftarrow \theta + \eta \nabla_\theta\mathbb E_{\pi_\theta} G(\tau)$$
> 5. fin du pour

En pratique on réalise en fait une version un petit peu plus efficace du VPG à l'aide de _batching_ : on collecte un _batch_ de trajectoires et on applique ce même algorithme au batch tout entier (les seules différences sont qu'on normalise en fonction du batch entier et que $L$ vaut le cardinal du batch et non $1$). 

Cela se traduit donc dans le code par les deux boucles ``for`` dans ``main()`` à la place d'une seule. 

### Calcul des récompenses
Pour rappel la récompense d'une trajectoire se calcule comme suit : 
$$G(\tau) = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$$
Son implémentation (dans `rewards.c`) est :
```c
double* discountedRewards(double* rewards, double gamma, int steps) {
    double* G = calloc(steps, sizeof(double));
    assert(G != NULL);
    
    G[steps - 1] = rewards[steps - 1];
    for (int t = steps - 2; t >= 0; t--) {
        G[t] = rewards[t] + gamma * G[t+1];
    }

    return G;
}
```
La fonction `rewards` est définie plus haut dans `rewards.c` et renvoie un `double` en fonction de deux états conséctutifs et de flags. 

> NB : En fait on n'appelle pas directement `discountedRewards` dans le code mais la fonction définie juste après `normRewards` qui normalise les récompenses sur le batch entier. Elle set détaillée plus bas dans la section sur la normalisation. 

### Calcul du gradient
La formule du gradient d'un paramètre $\theta$ est : 
$$\nabla_\theta\mathbb E_{\pi_\theta} G(\tau) = \frac 1 L \sum_\tau\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G(\tau)$$
où $\mathbb E$ représente l'espérance, $G(\tau)$ la récompense de la trajectoire $\tau$, $L$ le nombre de trajectoire et $\pi_\theta$ la politique. 

**Démonstration :**
$$
\begin{align*}
  \nabla_\theta \mathbb E_{\pi_\theta} G(\tau) &= \nabla_\theta \sum_\tau P(\tau \mid \theta) G(\tau) \\
  &= \sum_\tau \nabla_\theta P(\tau\mid\theta) G(\tau) \\
  &= \sum_\tau P(\tau\mid\theta)\frac{\nabla_\theta P(\tau\mid\theta)}{P(\tau\mid\theta)} G(\tau) \\
  &= \sum_\tau P(\tau\mid\theta) \nabla_\theta \log P(\tau\mid\theta) G(\tau) \\
  &= \mathbb E_{\pi_\theta} (\nabla_\theta \log P(\tau\mid\theta) G(\tau))
\end{align*}
$$
Calculons $\nabla_\theta\log P(\tau\mid\theta)$ : 
$$
\begin{align*}
  \nabla_\theta\log P(\tau\mid\theta) &= \nabla_\theta \log \left( p(s_0) \prod_{t=0}^{T-1} p(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t\mid s_t) \right) \\
  &= \nabla_\theta \left( \log p(s_0) + \sum_{t=0}^{T-1} \log p(s_{t+1} \mid s_t, a_t) + \log \pi_\theta(a_t\mid s_t) \right) \\
  &= \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\end{align*}
$$
On injecte : 
$$\nabla_\theta \mathbb E_{\pi_\theta} G(\tau) = \mathbb E_{\pi_\theta} \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G(\tau) \right)$$
Par méthode de Monte-Carlo on obtient finalement : 
$$\nabla_\theta\mathbb E_{\pi_\theta} G(\tau) = \frac 1 L \sum_\tau\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G(\tau)$$

Ce calcul est implémenté dans la fonction `backpropagation` dans le fichier `policy.c`, mais n'est pas fait d'un coup à cause des différentes optimisations (normalisations, bonus d'entropie, adam). 

Voici les différentes étapes du calcul :

1. On dérive le logarithme de softmax sur les logits (les valeurs obtenues avant passage au softmax). \
Voici la formule de la dérivée : 
$$\frac{\partial \log (\text{softmax}(z_i))}{\partial \text{logits}(z_j)} = 
\begin{cases} 
  1 - \text{softmax}(z_i) \quad\text{si } i = j\\ 
  -\text{softmax}(z_i) \quad\quad\text{si } i\neq j 
\end{cases}$$
$\quad\quad\space$ Dans le code cela se traduit par la ligne : 
```c
gadv = ((k == actionIndex) ? 1.0 : 0.0) - probs[k]` puis `dlogits[k] = gadv * G[t]
```

2. On multiplie par $G(\tau)$ (on aurait pu le faire plus tard et d'abord multiplier par $\frac{\partial \text{logits}}{\partial\theta}$) :
```c
dlogits[k] = gadv * G[t]
```

3. On dérive ensuite la couche de sortie (c'est simplement une couche dense, donc la dérivée est celle d'une fonction affine, on passe les détails) : 
```c
dWout[j][k] += h_t[j] * dlogits[k]
```
```c
dBout[k] += dlogits[k]
```

4. On dérive les paramètres de la couche de sortie sur l'état "caché" (traduction littérale pour hidden state) :
```c
dh[j] += \sum_k Wout[j][k] * dlogits[k]`
```

5. Finalement on dérive à travers chaque porte du LSTM. \
Les tableaux `f`, `i`, `g` et `o` sont les sorties des portes. `z` est l'entrée concaténée avec l'état "caché" (hidden state) précédent. `h` est l'état caché (qui est en fait la mémoire "court-terme"), `c` le "cell state" (soit la mémoire "long-terme") et `cprev` le cell state précédent. \
Les blocs de code effectuant la dérivée sont :
```c
do = dh * tanh(c_t);
d_o_pre = do * o * (1 - o);
```
```c
dc = dh * o * (1 - tanh(c)^2) + dc_next
```
```c
df = dc * c_{t-1}`;
d_f_pre = df * f * (1 - f);
```
```c
di = dc * g;
d_i_pre = di * i * (1 - i);
```
```c
dg = dc * i;
d_g_pre = dg * (1 - g^2);
```

6. Finalement chaque paramètre $\theta$ est mis-à-jour selon la montée de gradient donnée par la formule : 
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathbb E_{\pi_\theta} G(\tau)$$
$\quad\quad\space$ Pour les biais :
```c
dBf += d_f_pre;
dBi += d_i_pre;
dBc += d_g_pre;
dBo += d_o_pre;
```
$\quad\quad\space$ Pour les poids :
```c
dWf[a][j] += z[t][a] * d_f_pre[j];
dWi[a][j] += z[t][a] * d_i_pre[j];
dWc[a][j] += z[t][a] * d_g_pre[j];
dWo[a][j] += z[t][a] * d_o_pre[j];
```

## LSTM

## Softmax Temperature

## Epsilon-greedy

## Entropy Bonus

## Normalization

## Adam