> **ATTENTION** : GitHub gère mal MathJax (qui permet d'écrire du LaTeX dans le markdown), donc il est recommandé de lire le README localement ou avec un outil supportant MathJax.

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
gadv = ((k == actionIndex) ? 1.0 : 0.0) - probs[k]
```

2. On multiplie par $G(\tau)$ (on aurait pu le faire plus tard et d'abord multiplier par $\frac{\partial \text{logits}}{\partial\theta}$) :
```c
dlogits[k] = gadv * G[t]
```

3. On dérive ensuite la couche de sortie (c'est simplement une couche dense, donc la dérivée est celle d'une fonction affine, on passe les détails) : 
```c
for (int j = 0; j < H; j++) { 
  for (int k = 0; k < O; k++) dWout[j][k] += h[t][j] * dlogits[k]; 
}
```
```c
for (int k = 0; k < O; k++) dBout[k] += dlogits[k];
```

4. On dérive les paramètres de la couche de sortie sur l'état "caché" (traduction littérale pour hidden state) :
```c
for (int j = 0; j < H; j++) { 
  double s = 0.0; 
  for (int k = 0; k < O; k++) s += network->Wout[j][k] * dlogits[k]; 
  dh[j] += s; 
}
```

5. Finalement on dérive à travers chaque porte du LSTM (leur rôle/fonctionnement est détaillé plus tard dans la partie LSTM). \
Les tableaux `f`, `i`, `g` et `o` sont les sorties des portes. `z` est l'entrée concaténée avec l'état "caché" (hidden state) précédent. `h` est l'état caché (qui est en fait la mémoire "court-terme"), `c` le "cell state" (soit la mémoire "long-terme") et `cprev` le cell state précédent. \
Les blocs de code effectuant la dérivée sont :
```c
for (int j = 0; j < H; j++) do_vec[j] = dh[j] * tanh(c[t][j]);
for (int j = 0; j < H; j++) d_o_pre[j] = do_vec[j] * o[t][j] * (1.0 - o[t][j]);
```
```c
for (int j = 0; j < H; j++) dc[j] = dh[j] * o[t][j] * (1.0 - tanh(c[t][j]) * tanh(c[t][j])) + dc_next[j];
```
```c
for (int j = 0; j < H; j++) df[j] = dc[j] * cprev[t][j];
for (int j = 0; j < H; j++) d_f_pre[j] = df[j] * f[t][j] * (1.0 - f[t][j]);
```
```c
for (int j = 0; j < H; j++) di_vec[j] = dc[j] * g[t][j];
for (int j = 0; j < H; j++) d_i_pre[j] = di_vec[j] * i[t][j] * (1.0 - i[t][j]);
```
```c
for (int j = 0; j < H; j++) dg[j] = dc[j] * i[t][j];
for (int j = 0; j < H; j++) d_g_pre[j] = dg[j] * (1.0 - g[t][j] * g[t][j]);
```

6. Finalement chaque paramètre $\theta$ est mis-à-jour selon la montée de gradient donnée par la formule : 
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathbb E_{\pi_\theta} G(\tau)$$
$\quad\quad\space$ Pour les biais :
```c
for (int j = 0; j < H; j++) { 
  dBf[j] += d_f_pre[j]; 
  dBi[j] += d_i_pre[j]; 
  dBc[j] += d_g_pre[j]; 
  dBo[j] += d_o_pre[j]; 
}
```
$\quad\quad\space$ Pour les poids :
```c
for (int a = 0; a < Z; a++) { 
  for (int j = 0; j < H; j++) {
    dWf[a][j] += z[t][a] * d_f_pre[j];
    dWi[a][j] += z[t][a] * d_i_pre[j];
    dWc[a][j] += z[t][a] * d_g_pre[j];
    dWo[a][j] += z[t][a] * d_o_pre[j];
  } 
}
```

## LSTM
La politique utilisée est un LSTM (Long Short-Term Memory) qui est le réseau de neurones récurrent (RNN) le plus couramment utilisé. Son intérêt est son système de "mémoire" qui permet de mieux gérer les dépendances à long terme. \
Il y a quelques années c'était le standard pour les modèles de textes comme la traduction mais aussi les LLMs avant l'arrivée des Transformers, ce qui illustre bien son efficacité. 

On donne trois schémas pour illustrer le fonctionnement d'un LSTM, le premier pour mettre en évidence le rapport avec le thème "cycles et boucles" du TIPE et les deux autres pour détailler le fonctionnement du LSTM. 

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="Figure 1" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" alt="Figure 2" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*qToyitOZkf7Nhvr1LwxWgQ.png" alt="Figure 3" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">

Comme la figure 3 l'illustre, un LSTM est constitué de quatre "portes" (gates) que l'on nommera : la porte F (forget gate), la porte I (input gate), la porte G (candidate gate) et la porte O (output gate).

Notons de plus qu'il y a deux états intrinsèques au LSTM que l'on nommera état caché (pour hidden state) noté $h$ et état cellule (pour cell state) noté $c$. Ils représentent en fait respectivement la mémoire "court-terme" et la mémoire "long-terme" du LSTM.

Chaque porte a donc ses propres paramètres (poids et biais), d'où notre implémentation dans `struct.h` : 
```c
typeshit struct LSTM {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* hiddenState;
    double* cellState;

    double** Wf;
    double** Wi;
    double** Wc;
    double** Wo;
    double** Wout;

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
    double* Bout;
} LSTM;
```
Remarquons qu'on a les paramètres de la couche de sortie `Wout` et `Bout` en plus : nous avons choisit de rajouter une couche dense après le LSTM pour faire la sortie de la politique, ce qui permet notamment d'avoir un tableau avec une taille en sortie égale au nombre d'actions possibles (on peut donc directement appliquer un softmax dessus pour obtenir une distribution de probabilité sur les actions).

> **NB :** La structure présente dans `struct.h` possède d'autre champs pour l'optimiseur Adam que nous détaillerons après, ainsi que pour stocker des informations utiles lors de la backpropagation (comme les probabilités). 

Détaillons à présent le fonctionnement de chaque porte du LSTM : 
### Porte F (Forget Gate)
La porte F permet de décider quelles informations de l'état cellule (mémoire à long terme) doivent être oubliées de manière à faire de la place pour de nouvelles informations. \
Voici son équation : 
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + B_f)$$
où $\sigma$ est la fonction sigmoïde : 
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
et où $[h_{t-1}, x_t]$ représente la concaténation de l'état caché précédent $h_{t-1}$ et de l'entrée actuelle $x_t$.

Son implémentation dans `policy.c` est : 
```c
double* forgetGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bf[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wf[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}
```

### Porte I (Input Gate)
La porte I permet de décider quelles nouvelles informations doivent être ajoutées à l'état cellule (mémoire à long terme) en travaillant avec la porte G (détaillée juste en dessous). \
Voici son équation : 
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + B_i)$$
Son implémentation dans `policy.c` est :
```c
double* inputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bi[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wi[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}
```

### Porte G (Candidate Gate)
La porte G crée un vecteur de candidats qui pourraient être ajoutés à l'état cellule (mémoire à long terme) qui seront en fait sélectionnés par la porte I. \
Voici son équation :
$$g_t = \tanh(W_c \cdot [h_{t-1}, x_t] + B_c)$$
Notons donc l'équation de l'état cellule au passage :
$$c_t = f_t \cdot c_{t-1} + i_t * g_t$$
Son implémentation dans `policy.c` est :
```c
double* candidateGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bc[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wc[j][i];
        }
        result[i] = tanh(result[i]);
    }

    return result;
}
```

### Porte O (Output Gate)
La porte O filtre l'état cellule (mémoire à long terme) pour décider quelles parties doivent être envoyées à l'état caché (mémoire à court-terme) et donc à la sortie du LSTM. \
Voici son équation :
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + B_o)$$
On peut donc noter l'équation de l'état caché :
$$h_t = o_t \cdot \tanh(c_t)$$
Son implémentation dans `policy.c` est :
```c
double* outputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bo[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wo[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}
```

## Softmax Temperature
Un dilemme classique en RL est celui de l'exploration vs l'exploitation. En début d'entraînement il faut favoriser l'exploration, c'est pourquoi on utilise un système de température dans le softmax pour "lisser" les probabilités et favoriser l'exploration. \
La formule du softmax avec température est la suivante :
$$\text{softmax}(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
où $T$ est la température. \
Lorsque $T$ est grand, les probabilités sont plus uniformes (favorisant l'exploration), tandis que lorsque $T$ est petit, les probabilités sont plus concentrées sur les actions avec les plus hauts logits (favorisant l'exploitation).

Cela se traduit dans le code par l'argument `temperature` dans la fonction `forward` où est implémenté le softmax dans `policy.c` :
```c
  double sum = 0.0;
  for (int k=0; k<network->outputSize; k++) {
      network->probs[k] = exp(network->logits[k] / temperature);
      sum += network->probs[k];
  }
  for (int k=0; k<network->outputSize; k++) {
      network->probs[k] /= sum;
  }
```

## Epsilon-greedy

## Entropy Bonus

## Normalization

## Adam