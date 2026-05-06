# Resume des modifications ajoutees

Ce depot est maintenant organise en deux couches coherentes.

## 1. Noyau initial du projet

Le coeur experimental reste strictement base sur les notebooks suivants :

- `notebook1_setup.ipynb`
- `notebook2_forward_process.ipynb`
- `notebook3_architecture.ipynb`
- `notebook4_training.ipynb`
- `notebook5_evaluation.ipynb`
- `notebook6_writeup.ipynb`

Ces notebooks portent l'etude fondamentale : influence du bruit gaussien, uniforme et laplacien sur un pipeline de diffusion controle.

## 2. Extension appliquee : Battery Sentinel

Les notebooks ajoutes prolongent directement cette etude vers un systeme interpretable de suivi batterie :

- `notebook7_battery_sentinel_data.ipynb`
- `notebook8_battery_sentinel_twin.ipynb`
- `notebook9_battery_sentinel_router.ipynb`
- `notebook10_battery_sentinel_dashboard.ipynb`

Le lien avec la premiere partie est volontairement explicite :

- `gaussian` -> variation normale du systeme
- `uniform` -> telemetrie bornee, quantification, resolution limitee
- `laplace` -> chocs impulsifs, anomalies rares, deviations brusques

## Role de chaque notebook Battery Sentinel

### Notebook 7

- generation de sessions batterie simulees ;
- construction explicite des trois regimes d'incertitude ;
- sauvegarde du dataset de travail et des figures d'exemples.

### Notebook 8

- apprentissage d'un jumeau predictif nominal ;
- evaluation de ce jumeau sur les trois regimes ;
- sauvegarde du modele et des courbes d'entrainement.

### Notebook 9

- calcul des residus du jumeau ;
- routage des residus vers les regimes gaussien, uniforme ou laplacien ;
- production d'actions interpretablees : surveillance normale, demande de meilleure telemetrie, ou alerte d'inspection.

### Notebook 10

- synthese complete entre l'etude diffusionnelle initiale et le prototype batterie ;
- creation du tableau de bord final et des fichiers de synthese.

## Nettoyage effectue

Les anciens notebooks d'extension qui ne correspondaient plus a cette direction ont ete retires :

- `notebook7_iterative_forward_ablation.ipynb`
- `notebook8_cross_dataset_validation.ipynb`

Les deux services conserves sont :

- `:9000` pour l'explorateur local
- `:9001` pour les slides

## Utilite pour le projet

La premiere couche montre que les trois distributions de bruit n'ont pas le meme comportement dans un cadre generatif controle.

La seconde couche reutilise cette meme decomposition comme logique de decision dans un systeme applique. Le projet ne se limite donc plus a une comparaison de distributions ; il devient un prototype de routage d'incertitude interpretable pour le suivi batterie.
