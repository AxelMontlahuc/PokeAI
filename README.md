# Projet
Le but de ce TIPE est de créer un réseau de neurones capable de jouer à Pokémon Emeraude en C sans utiliser de librairies de Machine Learning. 

Pour cela nous utilisons mGBA comme émulateur et grâce à la librairie winsock (seule librairie non-standard utilisée) nous communiquons avec un script Lua dans mGBA pour récupérer l'état du jeu et envoyer des commandes.

Quand à l'agent, c'est un VPG (Vanilla Policy Gradient) utilisant pour politique un LSTM (Long Short-Term Memory), mettant en évidence les cycles et boucles implémenté de zéro en C.

# Utilisation
1. Cloner le dépôt.
2. Compiler le projet avec : 
```sh
~$ make
```
3. Placer une ROM de Pokémon Emeraude (version FR) nommée `pokemon.gba` dans le dossier `ROM`.
4. Lancer mGBA avec le script Lua `mGBASocketServer.lua` chargé.
5. Lancer l'agent avec : 
```shsh
~$ ./bin/agent
```

# Structure
- `mGBA-interface` : Librairie en C pour communiquer avec mGBA : appuyer sur des touches, lire la map. Pour plus d'informations, voir le [README](mGBA-interface/README.md).
- `mGBASocketServer.lua` : Script Lua utilisant l'API de mGBA pour faire le pont avec ``mGBA-interface`` à travers un socket.
- `src` : Code source de l'agent en C :
  - `agent` : Boucle principale de l'agent, gestion des épisodes, interaction avec l'environnement.
  - `policy` : Implémentation du LSTM et du VPG. 
  - `struct` : Définitions, initialisation et libération des structures de données.
  - `state` : Fonctions liées à l'état du jeu. 
  - `func` : Fonctions auxiliaires. 