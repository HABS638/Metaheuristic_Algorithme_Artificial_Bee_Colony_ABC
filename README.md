# 🐝 Artificial Bee Colony (ABC) Algorithm: Python vs C++

Ce projet met en œuvre et compare l’algorithme **Artificial Bee Colony (ABC)** en **Python** et **C++**.  
L’objectif est d’évaluer les performances des deux implémentations sur plusieurs fonctions de benchmark classiques : **Rosenbrock**, **Rastrigin**, et **Ackley**.

---

## 📘 Description du projet

L’**algorithme ABC** est une méthode d’optimisation inspirée du comportement des colonies d’abeilles dans la recherche de nourriture.  
Chaque abeille représente une solution candidate et interagit avec les autres pour trouver l’optimum global.

Le projet comporte :
- Une **implémentation Python** simple et lisible, facilitant l’expérimentation et l’analyse.
- Une **implémentation C++** plus performante, adaptée aux calculs intensifs.
- Une **comparaison expérimentale** entre les deux approches à l’aide du test statistique de **Wilcoxon**.

---

## ⚙️ Fonctionnalités principales

- Simulation des trois phases de l’algorithme : abeilles **employées**, **spectatrices**, et **éclaireuses**.  
- Exécution sur des fonctions de test non convexes :
  - Rosenbrock
  - Rastrigin
  - Ackley  
- Comparaison automatique Python vs C++ (moyenne, écart-type, test de Wilcoxon).  
- Visualisation graphique des performances et convergence.

---

## 🧩 Organisation du dépôt

| Dossier | Contenu |
|----------|----------|
| `Python/` | Code source en Python (`codeABCPython.py`) |
| `C++/` | Code source en C++ (`main.cpp`, `OriginalABC.cpp`, `OriginalABC.h`) |
| `Résultats/` | Graphiques et résultats comparatifs |
| `Rapport/` | Rapport PDF complet expliquant l’algorithme et les expérimentations |

