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
4. Entraîner le modèle avec : 
```sh
~$ ./train.sh <number of workers>  # Sur Linux
~$ .\train.bat <number of workers> # Sur Windows
```

> **Note :** Nous utilisons un système de "learner" et de "worker" pour accélérer l'apprentissage : le "learner" est responsable de l'entraînement du modèle (c'est lui qui fait la backpropagation) puis qui met à jour le modèle sauvegardé dans `checkpoints/` tandis que le "worker" charge le dernier modèle sauvegardé dans `checkpoint/` et l'utilise pour collecter des trajectoires qu'il fournit au "learner" dans `queue/`. \
> Cela permet donc de paralléliser la collecte de données (qui est la phase la plus lente car elle dépend de l'émulateur), ce qui accélère donc l'apprentissage. \
> Il faut donc adapter le nombre de "worker" en fonction de la puissance de l'ordinateur utilisé, généralement, lancer autant de "worker" que de threads (deux fois le nombre de coeurs du CPU la plupart du temps) disponibles est ce qui fonctionne le mieux. \
> Ce système se révèle d'autant plus efficace quand on entraîne le modèle sur un serveur (où l'on peut avoir 96 coeurs facilement par exemple). 

# Structure
- `bin` : Dossier des exécutables.
- `build` : Dossier des fichiers objets compilés.
- `ROM` : Dossier où placer la ROM de Pokémon Emeraude et éventuellement des savestates (sauvegardes).
- `checkpoints` : Dossier où seront sauvegardés les checkpoints (sauvegardes du modèle).
- `gba` : Dossier contenant le code pour l'émulateur. 
- `locks` : Dossier où seront placés de fichiers `.lock` pour éviter que plusieurs agents n'utilisent le même port.
- `logs` : Dossier où seront placés les fichiers de logs des agents.
- `makefile` : Fichier de compilation.
- `queue` : Dossier où seront placés les fichiers `.traj` représentant des trajectoires collectées par les agents et à utiliser pour l'apprentissage.
- `screen` : Dossier où seront placés les screenshots de l'émulateur pour que l'agent puisse lire l'état du jeu.
- `src` : Code source de l'agent en C :
  - `agent` : Boucle principale de l'agent, gestion des épisodes, interaction avec l'environnement.
  - `policy` : Implémentation du LSTM et du VPG. 
  - `state` : Fonctions liées à l'état/trajectoires du jeu. 
  - `checkpoint.c` : Sauvegarde et restauration du modèle complet. 

# Documentation
Nous allons ici rentrer un peu plus en détail sur l'implémentation et les mathématiques derrière l'agent en détaillant chaque partie/améliorations du modèle. \
Voici le plan de ce que nous allons détailler :

1. Proximal Policy Optimization - PPO
2. Long-Short Term Memory - LSTM (politique du PPO)
3. Système de température
4. Bonus d'entropie
5. Normalisation
6. Adam

## Proximal Policy Optimization
Le Proximal Policy Optimization (PPO) est un algorithme d'apprentissage par renforcement (RL) qui améliore la stabilité et l'efficacité de l'entraînement par rapport à ses prédecesseurs (comme le VPG). \
L'idée derrière le PPO est de limiter la mise à jour de la politique (la mise à jour dans la montée de gradient) pour éviter des changements trop brusques qui pourraient dégrader les performances de l'agent. Si cela ne paraît pas être un grand changement, c'est en fait extrêmement efficace en pratique.

> Notons que nous sommes directement passés de du VPG au PPO là où on aurait pu construire un modèle A2C (Advantage Actor-Critic) intermédiaire (ou bien en fait tout autre Actor-Critic). Nous allons donc détailler ce qu'est un Actor-Critic, qui est une technique classique en RL mais qui n'est pas propre au PPO.

Voici le pseudo-code de l'algorithme PPO (d'après [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)) : 
> _Entrée_ : Vecteur $\theta_0$ représentant les paramètres initiaux de la politique notée $\pi_\theta$, vecteur $\phi_0$ représentant les paramètres initiaux de la fonction V (pour _value function_) notée $V_\phi$. 
> 1. pour $k = 0, 1, 2, \ldots$ faire
> 2. $\quad$ Collecter un ensemble de trajectoires $\{\tau_i\}$ en exécutant la politique $\pi_{\theta_k}$ dans l'environnement.
> 3. $\quad$ Calculer les récompenses (équilibrées) $G(\tau_i)$ pour chaque trajectoire $\tau_i$.
> 4. $\quad$ Calculer les avantages estimés $\hat A_t$ en utilisant la fonction V actuelle $V_{\phi_k}$.
> 5. $\quad$ Mettre à jour les paramètres de la politique $\theta$ en _clippant_ les gradients de la politique :
> $$\theta \leftarrow \arg\max_\theta \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \min\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \hat A_t(s_t, a_t), g(\varepsilon, \hat A_t^{\pi_{\theta_k}}(s_t, a_t))\right)$$
> 6. $\quad$ Mettre à jour les paramètres de la fonction V $\phi$ en minimisant la perte de la fonction V :
> $$\phi \leftarrow \arg\min_\phi \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \left(V_\phi(s_t) - G(\tau)_t\right)^2$$
> 7. fin du pour

### Calcul des avantages
Le calcul des avantages n'est pas une technique propre au PPO mais une technique utilisée dans les algorithmes dits actor-critic pour réduire la variance des gradients : on utilise une fonction V pour estimer la valeur d'un état et on soustrait cette valeur aux récompenses pour obtenir les avantages (en fait l'avantage représente la "surperformance" de l'agent par rapport à la fonction V). 

On entraîne donc une fonction V en plus de la politique. Concrètement la fonction V calcule la récompense attendue à partir d'un état en suivant la politique actuelle. \
Améliorer l'avantage revient donc à améliorer la politique puisqu'on aurait plus de récompenses que prévu.

> L'intuition derrière le calcul des avantages est la suivante : dans un environnement comme Pokémon, les récompenses sont assez rares (l'agent va devoir faire beaucoup d'actions, de l'ordre d'une centaine dans notre cas avant d'obtenir une récompense). \
> La politique toute seule va donc avoir du mal à comprendre quelles actions sont bonnes ou mauvaises car la récompense est très éloignée dans le temps. En entraînant une fonction V, on entraîne en fait un modèle qui comprend mieux quand est-ce que l'agent va recevoir des récompenses, ce qui va donc permettre de mieux guider la politique (c'est comme si la fonction V donnait des récompenses intermédiaires à la politique en pratique (attention, en théorie pas du tout)). \
> Si la fonction V permet de réduire énormément la variance des gradients, elle a quand même un inconvénient : elle introduit du biais dans les gradients (ici il ne faut pas prendre biais au sens mathématique mais au sens "l'estimation n'est pas parfaite"). Si elle est bien entraînée en revanche et si l'obtention des récompenses a bien été réfléchie, ce biais devient alors négligeable (surtout comparé au gain dû à la réduction de variance).

La méthode la plus commune pour le calcul d'avantage est le GAE (Generalized Advantage Estimation) dont le but est de trouver un compromis entre biais et variance en utilisant deux hyperparamètres : le facteur  $\gamma$ (c'est le même que pour le VPG) et le paramètre $\lambda$ (qui aide à contrôler le ratio biais-variance). 

Voici les équations du GAE :
$$\hat A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}$$
où
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
Cependant dans la pratique, on calcule l'avantage récursivement, d'où la formule (plus simple!) : 
$$\hat A_t = \delta_t + \gamma \lambda \hat A_{t+1}$$

Voici l'implémentation du GAE dans `rewards.c` : 
```c
void computeGAE(double* rewards, double* values, int steps, double gamma, double lambda, double* out_advantages, double* out_returns) {
    double gae = 0.0;

    for (int t=steps-1; t>=0; t--) {
        double v = values[t];
        double v_next = (t + 1 < steps) ? values[t+1] : 0.0;

        double delta = rewards[t] + gamma * v_next - v;

        gae = delta + gamma * lambda * gae;
        
        out_advantages[t] = gae;
        out_returns[t] = out_advantages[t] + v;
    }
}
```

### Clipping
Le clipping est la composante principale de l'algorithme PPO : c'est l'amélioration qu'apporte le PPO par rapport à ses prédecesseurs. \
Il vise à maximiser la performance de la politique tout en limitant les mises à jour trop importantes : on "clip" le ratio entre la nouvelle politique et l'ancienne pour éviter des changements trop brusques.

On redonne la formule : 
$$\theta \leftarrow \arg\max_\theta \frac{1}{|\mathcal D_k|T} \sum_{\tau \in \mathcal D_k} \sum_{t=0}^{T} \min\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \hat A_t(s_t, a_t), g(\varepsilon, \hat A_t^{\pi_{\theta_k}}(s_t, a_t))\right)$$
La fonction $g$ est définie comme suit :
$$g(\varepsilon, A) = \begin{cases}
(1 + \varepsilon) A & \text{si } A \geq 0 \\
(1 - \varepsilon) A & \text{si } A < 0
\end{cases}$$
où $\varepsilon$ est un paramètre (typiquement entre $0.1$ et $0.3$) qui contrôle à quel point on "clip" les gradients.

L'implémentation se trouve dans `policy.c`. 

### Mise à jour de la fonction V
Les coefficients sont directement stockés dans la structure du LSTM, et on les met à jour de manière classique en minimisant la MSE. \
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

## Système de température
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
    double e = exp((network->logits[k] - maxlog) / temperature);
    network->probs[k] = e;
    sum += e;
}
for (int k = 0; k < O; k++) network->probs[k] /= sum + NUM_EPS;
```
On fait donc baisser la température au fur et à mesure de l'entraînement pour favoriser l'exploitation une fois que l'agent a suffisamment exploré l'environnement. 

Voici la ligne responsable de la mise-à-jour de la température : 
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
void entropyBonus(const double* probs, int O, double coeff, double* dlogits) {
    double totalH = 0.0; // En fait c'est -H

    for (int k = 0; k < O; k++) {
        totalH += probs[k] * log(probs[k] + NUM_EPS);
    }
    
    for (int k = 0; k < O; k++) {
        double gradH = probs[k] * (totalH - log(probs[k] + NUM_EPS));
        dlogits[k] += coeff * gradH;
    }
}
```
```c
entropyBonus(probs_now, O, ENTROPY_COEFF, dlogits);
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

    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double d = G[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;

    double std_deviation = sqrt(variance) + STD_EPS;

    for (int i = 0; i < n; i++) G[i] = (G[i] - mean) / std_deviation;
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
    double* Wv

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
    double* Bout;
    double Bv;

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
    double* Wv_m;
    double* Wv_v;

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
    double Bv_m;
    double Bv_v;
} LSTM;
```

Adam est implémenté dans la fonction `backpropagation` à l'aide de fonctions auxiliaires : 
```c
static inline void adam_update_scalar(double* param, double* m, double* v, double grad, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    double g = grad * scale;
    *m = beta1 * (*m) + (1.0 - beta1) * g;
    *v = beta2 * (*v) + (1.0 - beta2) * (g * g);
    double mhat = (*m) * inv_bc1;
    double vhat = (*v) * inv_bc2;
    *param += lr * (mhat / (sqrt(vhat) + eps));
}

static void adam_update_vector(double* P, double* M, double* V, const double* G, int n, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    for (int i = 0; i < n; i++) {
        adam_update_scalar(&P[i], &M[i], &V[i], G[i], lr, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    }
}

static void adam_update_matrix(double** P, double** M, double** V, double** G, int rows, int cols, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            adam_update_scalar(&P[r][c], &M[r][c], &V[r][c], G[r][c], lr, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
        }
    }
}
```
Puis on appelle ces fonctions pour chaque paramètre du LSTM : 
```c
const double beta1 = ADAM_BETA1;
const double beta2 = ADAM_BETA2;
const double eps = ADAM_EPS;
network->adam_t += 1;
double bc1 = 1.0 - pow(beta1, (double)network->adam_t);
double bc2 = 1.0 - pow(beta2, (double)network->adam_t);
double inv_bc1 = 1.0 / bc1;
double inv_bc2 = 1.0 / bc2;

adam_update_matrix(network->Wf, network->Wf_m, network->Wf_v, dWf, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_matrix(network->Wi, network->Wi_m, network->Wi_v, dWi, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_matrix(network->Wc, network->Wc_m, network->Wc_v, dWc, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_matrix(network->Wo, network->Wo_m, network->Wo_v, dWo, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_matrix(network->Wout, network->Wout_m, network->Wout_v, dWout, H, O, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);

adam_update_vector(network->Bf, network->Bf_m, network->Bf_v, dBf, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_vector(network->Bi, network->Bi_m, network->Bi_v, dBi, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_vector(network->Bc, network->Bc_m, network->Bc_v, dBc, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_vector(network->Bo, network->Bo_m, network->Bo_v, dBo, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_vector(network->Bout, network->Bout_m, network->Bout_v, dBout, O, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);

adam_update_vector(network->Wv, network->Wv_m, network->Wv_v, dWv, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
adam_update_scalar(&network->Bv, &network->Bv_m, &network->Bv_v, dBv, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
```