> **ATTENTION** : GitHub gère mal MathJax (qui permet d'écrire du LaTeX dans le markdown), donc il est recommandé de lire le README localement ou avec un outil supportant MathJax.

# Projet
Le but de ce TIPE est de créer un réseau de neurones capable de jouer à Pokémon Emeraude en C sans utiliser de librairies de Machine Learning, de manière à tout recoder de zéro pour mieux comprendre les maths derrière l'agent que nous allons construire. 

Pour cela nous utilisons mGBA comme émulateur et grâce à la librairie winsock/socket (seules librairies non-standard utilisées) nous communiquons avec un script Lua dans mGBA pour récupérer l'état du jeu et envoyer des commandes.

L'agent utilise commme algorithme le PPO (Proximal Policy Optimization) avec un LSTM (Long Short-Term Memory) comme politique, mettant en évidence les cycles et boucles.

# Utilisation
1. Cloner le dépôt :
```sh
~$ git clone https://github.com/AxelMontlahuc/PokeAI.git
```
2. Placer une ROM de Pokémon Emeraude (version FR) nommée `pokemon.gba` dans le dossier `ROM`.
3. Compiler le projet avec : 
```sh
~$ make
```
4. Lancer le programme responsable de l'apprentissage :
```sh
~$ ./bin/learner     # Sur Linux
~$ ./bin/learner.exe # Sur Windows
```
5. Lancer $N$ instances de mGBA avec le script Lua `mGBASocketServer.lua` chargé.
6. Lancer un agent pour chaque instance de mGBA :
```sh
~$ ./bin/agent 88XX     # Sur Linux
~$ ./bin/agent.exe 88XX # Sur Windows
```
où `88XX` est le port sur lequel l'instance de mGBA écoute (pour le premier agent mettre `8888`, pour le deuxième `8889`, etc).

# Structure
- `mGBASocketServer.lua` : Script Lua utilisant l'API de mGBA pour faire le pont avec `mGBA-interface` à travers un socket.
- `mGBA-interface` : Librairie en C pour communiquer avec mGBA : appuyer sur des touches et lire la map et autre données du jeu. Pour plus d'informations, voir le [README](mGBA-interface/README.md).
- `src` : Code source de l'agent en C :
  - `agent` : Boucle principale de l'agent, gestion des épisodes, interaction avec l'environnement.
  - `policy` : Implémentation du LSTM et du VPG. 
  - `state` : Fonctions liées à l'état/trajectoires du jeu. 
  - `checkpoint.c` : Sauvegarde et restauration du modèle complet. 
- `bin` : Dossier des exécutables.
- `build` : Dossier des fichiers objets compilés.
- `ROM` : Dossier où placer la ROM de Pokémon Emeraude et éventuellement des savestates (sauvegardes).
- `checkpoints` : Dossier où seront sauvegardés les checkpoints (sauvegardes du modèle).
- `locks` : Dossier où seront placés de fichiers `.lock` pour éviter que plusieurs agents n'utilisent le même port.
- `queue` : Dossier où seront placés les fichiers `.traj` représentant des trajectoires collectées par les agents et à utiliser pour l'apprentissage.

# Documentation
Nous allons ici rentrer un peu plus en détail sur l'implémentation et les mathématiques derrière l'agent en détaillant chaque caractéristique du modèle. \
Voici le plan des caractéristiques que nous allons détailler :

1. Proximal Policy Optimization
2. LSTM
3. Softmax Temperature
4. Bonus d'entropie
5. Normalisation
6. Adam

## Proximal Policy Optimization
Le Proximal Policy Optimization (PPO) est un algorithme d'apprentissage par renforcement (RL) qui améliore la stabilité et l'efficacité de l'entraînement par rapport à ses prédecesseurs (comme le VPG). \
L'idée derrière le PPO est de limiter la mise à jour de la politique (la mise à jour dans la montée de gradient) pour éviter des changements trop brusques qui pourraient dégrader les performances de l'agent. Si cela ne paraît pas être un grand changement, c'est en fait extrêmement efficace en pratique.

Voici le pseudo-code de l'algorithme PPO (d'après [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)) : 
> _Entrée_ : Vecteur $\theta_0$ représentant les paramètres initiaux de la politique notée $\pi_\theta$, vecteur $\phi_0$ représentant les paramètres initiaux de la fonction V (pour _value function_) notée $V_\phi$. 
> 1. pour $k = 0, 1, 2, \ldots$ faire
> 2. $\quad$ Collecter un ensemble de trajectoires $\{\tau_i\}$ en exécutant la politique $\pi_{\theta_k}$ dans l'environnement.
> 3. $\quad$ Calculer les récompenses (équilibrées) $G(\tau_i)$ pour chaque trajectoire $\tau_i$.
> 4. $\quad$ Calculer les avantages estimés $\hat A_t$ en utilisant la fonction V actuelle $V_{\phi_k}$.
> 5. $\quad$ Mettre à jour les paramètres de la politique $\theta$ en maximisant l'objectif PPO-Clip :
> $$\theta \leftarrow \arg\max_\theta \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \min\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \hat A_t(s_t, a_t), g(\varepsilon, \hat A_t^{\pi_{\theta_k}}(s_t, a_t))\right)$$
> 6. $\quad$ Mettre à jour les paramètres de la fonction V $\phi$ en minimisant la perte de la fonction V :
> $$\phi \leftarrow \arg\min_\phi \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \left(V_\phi(s_t) - G(\tau)_t\right)^2$$
> 7. fin du pour

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

### Calcul des avantages
Le calcul des avantages n'est pas une technique propre au PPO mais une technique utilisée dans les algorithmes dits actor-critic pour réduire la variance des gradients : on utilise une fonction V pour estimer la valeur d'un état et on soustrait cette valeur aux récompenses pour obtenir les avantages (en fait l'avantage représente la "surperformance" de l'agent par rapport à la fonction V). 

On entraîne donc une fonction V en plus de la politique. Concrètement la fonction V calcule la récompense attendue à partir d'un état en suivant la politique actuelle. \
Améliorer l'avantage revient donc à améliorer la politique puisqu'on aurait plus de récompenses que prévu.

La méthode la plus commune pour le calcul d'avantage est le GAE (Generalized Advantage Estimation) qui utilise une combinaison de plusieurs estimateurs d'avantages avec différents horizons temporels pour obtenir un compromis entre biais et variance. \

Voici l'implémentation du GAE dans `rewards.c` : 
```c
void computeGAE(
    const double* rewards,
    const double* values,
    int steps,
    double gamma,
    double gae_lambda,
    double* out_advantages,
    double* out_returns
) {
    double gae = 0.0;
    for (int t = steps - 1; t >= 0; t--) {
        double v_t = values[t];
        double v_tp1 = (t + 1 < steps) ? values[t + 1] : 0.0;
        double delta = rewards[t] + gamma * v_tp1 - v_t;
        gae = delta + gamma * gae_lambda * gae;
        out_advantages[t] = gae;
        out_returns[t] = out_advantages[t] + v_t;
    }
}
```

### Objectif PPO-Clip
L'objectif PPO-Clip est la composante principale de l'algorithme PPO. Il vise à maximiser la performance de la politique tout en limitant les mises à jour trop importantes : on "clip" le ratio entre la nouvelle politique et l'ancienne pour éviter des changements trop brusques.

On redonne la formule : 
$$\theta \leftarrow \arg\max_\theta \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \min\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \hat A_t(s_t, a_t), g(\varepsilon, \hat A_t^{\pi_{\theta_k}}(s_t, a_t))\right)$$
La fonction $g$ est définie comme suit :
$$g(\varepsilon, A) = \begin{cases}
(1 + \varepsilon) A & \text{si } A \geq 0 \\
(1 - \varepsilon) A & \text{si } A < 0
\end{cases}$$

L'implémentation se trouve dans `policy.c`. 

### Mise à jour de la fonction V
Les coefficients sont directement stockés dans la structure du LSTM, et on les met à jour de manière classique en minimisant la MSE. 

On ne s'attarde pas dessus (implémentation dans `policy.c`).

## LSTM
La politique utilisée est un LSTM (Long Short-Term Memory) qui est le réseau de neurones récurrent (RNN) le plus couramment utilisé. Son intérêt est son système de "mémoire" qui permet de mieux gérer les dépendances à long terme. \
Il y a quelques années c'était le standard pour les modèles de textes comme la traduction mais aussi les LLMs avant l'arrivée des Transformers, ce qui illustre bien son efficacité. 

On donne trois schémas pour illustrer le fonctionnement d'un LSTM, le premier pour mettre en évidence le rapport avec le thème "cycles et boucles" du TIPE et les deux autres pour détailler le fonctionnement du LSTM. 

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="Figure 1" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" alt="Figure 2" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*qToyitOZkf7Nhvr1LwxWgQ.png" alt="Figure 3" width="500" style="position: relative; left: 50%; transform: translateX(-50%);">

Comme la figure 3 l'illustre, un LSTM est constitué de quatre "portes" (gates) que l'on nommera : la porte F (forget gate), la porte I (input gate), la porte G (candidate gate) et la porte O (output gate).

Notons de plus qu'il y a deux états intrinsèques au LSTM que l'on nommera état caché (pour hidden state) noté $h$ et état cellule (pour cell state) noté $c$. Ils représentent en fait respectivement la mémoire "court-terme" et la mémoire "long-terme" du LSTM.

Chaque porte a donc ses propres paramètres (poids et biais), d'où notre implémentation dans `policy.h` : 
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

> **NB :** La structure présente dans `policy.h` possède d'autre champs pour l'optimiseur Adam que nous détaillerons après, ainsi que pour stocker des informations utiles lors de la backpropagation (comme les probabilités). 

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
int O = network->outputSize;
double maxlog = network->logits[0];
for (int k = 1; k < O; k++) {
    if (network->logits[k] > maxlog) maxlog = network->logits[k];
}

double sum = 0.0;
for (int k = 0; k < O; k++) {
    double z = (network->logits[k] - maxlog) / temperature;
    double e = exp(z);
    network->probs[k] = e;
    sum += e;
}
double inv = 1.0 / (sum + 1e-12);
for (int k = 0; k < O; k++) network->probs[k] *= inv;
```
On fait donc baisser la température au fur et à mesure de l'entraînement pour favoriser l'exploitation une fois que l'agent a suffisamment exploré l'environnement. 

Voici la ligne dans `agent.c` responsable de la mise-à-jour de la température : 
```c
temperature = fmax(1.0, 3.0 * pow(0.97, (double)episode));
```

## Bonus d'entropie
Une dernière méthode pour favoriser l'exploration est d'ajouter un bonus d'entropie. L'idée est d'encourager l'agent à maintenir une certaine diversité dans ses actions en récompensant les politiques plus "incertaines" (c'est-à-dire avec une entropie plus élevée). 

L'entropie $H$ d'une distribution de probabilité $P$ se calcule de la façon suivante :
$$H(P) = -\sum_{i} P(i) \log(P(i))$$

Ici on ajoute directement un bonus d'entropie au gradient de la politique qui devient donc : 
$$\nabla_\theta\mathbb E_{\pi_\theta} G(\tau) = \frac 1 L \sum_\tau\sum_{t=0}^{T-1} \left( \nabla_\theta \log \pi_\theta(a_t\mid s_t) G(\tau) + \beta \nabla_\theta H(\pi_\theta(\cdot\mid s_t)) \right)$$
Détaillons la dérivée de l'entropie :
$$
\begin{align*}
  \nabla_\theta H(\pi_\theta(\cdot\mid s_t)) &= \pi_\theta(\cdot\mid s_t) (\mathbb E_{\pi_\theta} (\log \pi_\theta(\cdot\mid s_t)) - \log \pi_\theta(i\mid s_t)) \\
  &= \pi_\theta(\cdot\mid s_t) \left( \left( \sum_k \pi_\theta(k\mid s_t) \log \pi_\theta(k\mid s_t) \right) - \log \pi_\theta(i\mid s_t) \right)
\end{align*}
$$

L'implémentation dans `policy.c` est donc la suivante : 
```c
static const double ENTROPY_COEFF = 0.01; // Coefficient beta
```
```c
void entropyBonus(const double* probs, int O, double coeff, double* dlogits) {
    double mean_logp = 0.0;
    for (int k = 0; k < O; k++) {
        double pk = probs[k];
        mean_logp += pk * log(pk + 1e-12);
    }
    for (int k = 0; k < O; k++) {
        double pk = probs[k];
        double ent_grad = pk * (mean_logp - log(pk + 1e-12));
        dlogits[k] += coeff * ent_grad;
    }
}
```
```c
entropyBonus(tr->probs[t], O, ENTROPY_COEFF, dlogits);
```

## Normalisation
Notre code utilise deux types de normalisation : la normalisation des récompenses et la normalisation du gradient. 

### Normalisation des récompenses
La normalisation des récompenses permet de stabiliser l'entraînement en s'assurant que les récompenses ont une moyenne nulle et un écart-type unitaire. Cela aide à éviter que les gradients ne deviennent trop grands ou trop petits, ce qui tuerait l'apprentissage.

Pour normaliser les récompenses, on utlilise la formule suivante :
$$G_{\text{normé}}(t) = \frac{G(t) - \hat G}{\sigma_G}$$
où $\hat G$ est la moyenne des récompenses et $\sigma_G$ est l'écart-type des récompenses.

Voici l'implémentation dans `rewards.c` : 
```c
void normRewards(double* G, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += G[i];
    mean /= (double)n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = G[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    double std = sqrt(var) + 1e-8;

    for (int i = 0; i < n; i++) G[i] = (G[i] - mean) / std;
}
```

### Normalisation des gradients
La normalisation des gradients permet d'avoir un apprentissage plus stable en évitant notamment le problème d'explosion des gradients (les gradients divergent). 

Ici on utilise le _L2 norm clipping_ qui consiste à calculer la norme 2 du gradient et à le redimensionner si elle dépasse un certain seuil $c$.

La formule de la norme $L2$ d'un vecteur $\theta$ est la suivante :
$$\|\theta\|_2 = \sqrt{\sum_i \theta_i^2}$$

Si $\|\nabla_\theta\|_2 > c$ alors on redimensionne le gradient comme suit :
$$\nabla_\theta \leftarrow \nabla_\theta \cdot \frac{c}{\|\nabla_\theta\|_2}$$

Voici l'implémentation dans  la fonction `backpropagation` de `policy.c` : 
```c
double clip = 1.0; 
double norm2 = 0.0;
for (int a = 0; a < Z; a++) { 
    for (int j = 0; j < H; j++) { 
        norm2 += dWf[a][j]*dWf[a][j]; 
        norm2 += dWi[a][j]*dWi[a][j]; 
        norm2 += dWc[a][j]*dWc[a][j]; 
        norm2 += dWo[a][j]*dWo[a][j]; 
    } 
}
for (int j = 0; j < H; j++) { 
    for (int k = 0; k < O; k++) norm2 += dWout[j][k]*dWout[j][k]; 
}
for (int j = 0; j < H; j++) norm2 += dBf[j]*dBf[j] + dBi[j]*dBi[j] + dBc[j]*dBc[j] + dBo[j]*dBo[j];
for (int k = 0; k < O; k++) norm2 += dBout[k]*dBout[k];

double norm = sqrt(norm2); 
double scale = (norm > clip) ? (clip / (norm + 1e-12)) : 1.0;
```
Puis chaque gradient est multiplié par `scale` : 
```c
g = dWf[a][j] * scale;
```
```c
g = dWi[a][j] * scale;
```
```c
g = dWc[a][j] * scale;
```
```c
g = dWo[a][j] * scale;
```
```c
double g = dWout[j][k] * scale;
```
```c
g = dBf[j] * scale;
```
```c
g = dBi[j] * scale;
```
```c
g = dBc[j] * scale;
```
```c
g = dBo[j] * scale;
```
```c
double g = dBout[k] * scale;
```

## Adam
L'entraînement du réseau de neurone est très long, notamment parce qu'il dépend de la vitesse du jeu dans mGBA qu'on arrive seulement à accélerer en x10. Pour accélérer l'entraînement, soit la convergence du modèle vers la politique optimale on utilise l'optimisateur Adam (Adaptive Moment Estimation). 

L'idée derrière Adam est d'ajuster dynamiquement le taux d'apprentisage pour chaque paramètre en fonction de ce qu'on appelle les moments : la moyenne (premier moment) et la variance (second moment) des gradients. \
En particulier ces deux moments viennent en fait de deux autres optimisateurs : le Momentum (pour la moyenne) et le RMSProp (pour la variance) que nous allons un peu détailler avant de passer à Adam.

On rappelle la formule de la montée de gradient standard :
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathbb E_{\pi_\theta} G(\tau)$$

### Momentum
Le Momentum accélère la convergence en ajoutant un terme en fonction de la moyenne des gradients passés à la montée de gradient. 

Voici sa formule : 
$$\theta \leftarrow \theta - \eta m_t$$
Avec $m_t$ la moyenne mobile des gradients calculée comme suit : 
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta \mathbb E_{\pi_\theta} G(\tau)$$
où $\beta_1$ est un paramètre (typiquement $0.9$). 

### RMSProp
Le RMSProp adapte le taux d'apprentissage en fonction de la variance des gradients passés.

Voici sa formule :
$$\theta \leftarrow \theta - \eta \frac{\nabla_\theta \mathbb E_{\pi_\theta} G(\tau)}{\sqrt{v_t} + \varepsilon}$$
Avec $v_t$ la moyenne mobile des carrés des gradients calculée comme suit :
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta \mathbb E_{\pi_\theta} G(\tau))^2$$
où $\beta_2$ est un paramètre (typiquement $0.9$) et $\varepsilon$ une petite constante pour éviter la division par zéro (typiquement $10^{-8}$).

### Adam
Finalement Adam combine les deux avec une petite subtilité : on corrige les biais initiaux des moments ($m_t$ et $v_t$) en les divisant par $(1 - \beta_1^t)$ et $(1 - \beta_2^t)$ respectivement pour qu'ils ne soient pas trop proches de zéro au début de l'entraînement : 
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad\text{et}\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Finalement la montée de gradient devient : 
$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Comme on doit stocker les gradients passés pour calculer $m_t$ et $v_t$, nous avons ajouté des champs dans la structure `LSTM` pour stocker ces valeurs (défini dans `policy.h`). Voici donc la structure complète : 
```c
typeshit struct LSTM {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* hiddenState;
    double* cellState;
    double* logits; // On est obligé de stocker ces valeurs
    double* probs;  // pour la backpropagation

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

    int adam_t; // Compteur de pas
    double** Wf_m; 
    double** Wf_v;
    double** Wi_m; 
    double** Wi_v;
    double** Wc_m; 
    double** Wc_v;
    double** Wo_m; 
    double** Wo_v;
    double** Wout_m; 
    double** Wout_v;

    double* Bf_m; 
    double* Bf_v;
    double* Bi_m; 
    double* Bi_v;
    double* Bc_m; 
    double* Bc_v;
    double* Bo_m; 
    double* Bo_v;
    double* Bout_m; 
    double* Bout_v;
} LSTM;
```

Adam est directement implémenté dans la fonction `backpropagation` de `policy.c` : 
```c
const double beta1 = 0.9;
const double beta2 = 0.999;
const double eps = 1e-8;
network->adam_t += 1;
double bc1 = 1.0 - pow(beta1, (double)network->adam_t);
double bc2 = 1.0 - pow(beta2, (double)network->adam_t);
double inv_bc1 = 1.0 / bc1;
double inv_bc2 = 1.0 / bc2;

for (int a = 0; a < Z; a++) {
    for (int j = 0; j < H; j++) {
        double g;
        
        g = dWf[a][j] * scale;
        network->Wf_m[a][j] = beta1 * network->Wf_m[a][j] + (1.0 - beta1) * g;
        network->Wf_v[a][j] = beta2 * network->Wf_v[a][j] + (1.0 - beta2) * (g * g);
        double mhat = network->Wf_m[a][j] * inv_bc1;
        double vhat = network->Wf_v[a][j] * inv_bc2;
        network->Wf[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
        
        g = dWi[a][j] * scale;
        network->Wi_m[a][j] = beta1 * network->Wi_m[a][j] + (1.0 - beta1) * g;
        network->Wi_v[a][j] = beta2 * network->Wi_v[a][j] + (1.0 - beta2) * (g * g);
        mhat = network->Wi_m[a][j] * inv_bc1;
        vhat = network->Wi_v[a][j] * inv_bc2;
        network->Wi[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
        
        g = dWc[a][j] * scale;
        network->Wc_m[a][j] = beta1 * network->Wc_m[a][j] + (1.0 - beta1) * g;
        network->Wc_v[a][j] = beta2 * network->Wc_v[a][j] + (1.0 - beta2) * (g * g);
        mhat = network->Wc_m[a][j] * inv_bc1;
        vhat = network->Wc_v[a][j] * inv_bc2;
        network->Wc[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
        
        g = dWo[a][j] * scale;
        network->Wo_m[a][j] = beta1 * network->Wo_m[a][j] + (1.0 - beta1) * g;
        network->Wo_v[a][j] = beta2 * network->Wo_v[a][j] + (1.0 - beta2) * (g * g);
        mhat = network->Wo_m[a][j] * inv_bc1;
        vhat = network->Wo_v[a][j] * inv_bc2;
        network->Wo[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
    }
}
for (int j = 0; j < H; j++) {
    for (int k = 0; k < O; k++) {
        double g = dWout[j][k] * scale;
        network->Wout_m[j][k] = beta1 * network->Wout_m[j][k] + (1.0 - beta1) * g;
        network->Wout_v[j][k] = beta2 * network->Wout_v[j][k] + (1.0 - beta2) * (g * g);
        double mhat = network->Wout_m[j][k] * inv_bc1;
        double vhat = network->Wout_v[j][k] * inv_bc2;
        network->Wout[j][k] += learningRate * (mhat / (sqrt(vhat) + eps));
    }
}

for (int j = 0; j < H; j++) {
    double g;
    g = dBf[j] * scale;
    network->Bf_m[j] = beta1 * network->Bf_m[j] + (1.0 - beta1) * g;
    network->Bf_v[j] = beta2 * network->Bf_v[j] + (1.0 - beta2) * (g * g);
    double mhat = network->Bf_m[j] * inv_bc1;
    double vhat = network->Bf_v[j] * inv_bc2;
    network->Bf[j] += learningRate * (mhat / (sqrt(vhat) + eps));

    g = dBi[j] * scale;
    network->Bi_m[j] = beta1 * network->Bi_m[j] + (1.0 - beta1) * g;
    network->Bi_v[j] = beta2 * network->Bi_v[j] + (1.0 - beta2) * (g * g);
    mhat = network->Bi_m[j] * inv_bc1;
    vhat = network->Bi_v[j] * inv_bc2;
    network->Bi[j] += learningRate * (mhat / (sqrt(vhat) + eps));

    g = dBc[j] * scale;
    network->Bc_m[j] = beta1 * network->Bc_m[j] + (1.0 - beta1) * g;
    network->Bc_v[j] = beta2 * network->Bc_v[j] + (1.0 - beta2) * (g * g);
    mhat = network->Bc_m[j] * inv_bc1;
    vhat = network->Bc_v[j] * inv_bc2;
    network->Bc[j] += learningRate * (mhat / (sqrt(vhat) + eps));

    g = dBo[j] * scale;
    network->Bo_m[j] = beta1 * network->Bo_m[j] + (1.0 - beta1) * g;
    network->Bo_v[j] = beta2 * network->Bo_v[j] + (1.0 - beta2) * (g * g);
    mhat = network->Bo_m[j] * inv_bc1;
    vhat = network->Bo_v[j] * inv_bc2;
    network->Bo[j] += learningRate * (mhat / (sqrt(vhat) + eps));
}

for (int k = 0; k < O; k++) {
    double g = dBout[k] * scale;
    network->Bout_m[k] = beta1 * network->Bout_m[k] + (1.0 - beta1) * g;
    network->Bout_v[k] = beta2 * network->Bout_v[k] + (1.0 - beta2) * (g * g);
    double mhat = network->Bout_m[k] * inv_bc1;
    double vhat = network->Bout_v[k] * inv_bc2;
    network->Bout[k] += learningRate * (mhat / (sqrt(vhat) + eps));
}
```