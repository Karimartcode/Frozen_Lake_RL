# Apprentissage par renforcement sécuritaire sur Frozen Lake
 
**IFT-7201 - Projet**
 
Étude de l'influence des dynamiques non-déterministes sur l'évitement de situations dangereuses en apprentissage par renforcement.
 
## Auteurs
- Abdelkarim Mouachiq — abdelkarim.mouachiq.1@ulaval.ca
- Saad ElGhissassi — saad.elghissassi.1@ulaval.ca
 
## Structure du projet
 
```
├── src/
│   ├── environment.py          # Configuration de l'environnement Frozen Lake
│   ├── visualize.py            # Utilitaires de visualisation
│   ├── train_baselines.py      # Entraînement DQN et PPO
│   ├── run_experiments.py      # Exécution des expériences baselines
├── report/
│   ├── report_preliminary.tex  # Rapport préliminaire
│   ├── arxiv.sty               # Style LaTeX
│   ├── references.bib          # Bibliographie
│   ├── figures/                # Figures du rapport
│   └── tables/                 # Tableaux du rapport
├── results/
│   ├── figures/                # Figures générées
│   └── *.json                  # Résultats bruts des expériences
└── requirements.txt
```
 
## Installation
 
```bash
pip install -r requirements.txt
```
 
## Exécution des expériences
 
```bash
# Expériences baselines (DQN + PPO)
python src/run_experiments.py
```
 
## Cas d'étude
 
| Cas | Grille | Trous | Ratio de danger |
|-----|--------|-------|-----------------|
| 1   | 4x4    | 1     | 6.25%           |
| 2   | 8x8    | 10    | 15.6%           |
| 3   | 8x8    | 17    | 26.6%           |
 
## Stratégies
 
- **DQN** (Deep Q-Network) — méthode basée sur la valeur
- **PPO** (Proximal Policy Optimization) — méthode basée sur la politique
