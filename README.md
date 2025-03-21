# Application de Traitement d'Image avec Streamlit

Cette application web interactive permet de traiter et transformer des images à l'aide de diverses techniques. Développée avec Python et Streamlit, elle offre une interface utilisateur intuitive pour appliquer des transformations visuelles sur vos images.

## Fonctionnalités

### Transformations de Base
- **Conversion en niveaux de gris** : Transforme une image colorée en noir et blanc
- **Rotation** : Tourne l'image selon un angle spécifié
- **Redimensionnement** : Modifie la taille de l'image
- **Symétrie** : Applique un effet miroir horizontal ou vertical

### Transformations Avancées
- **Ajustement de luminosité et contraste** : Modifie l'intensité lumineuse et le contraste
- **Flou gaussien** : Applique un effet de flou paramétrable
- **Détection de contours** : Met en évidence les bordures des objets dans l'image

### Transformation de Fourier
- Visualisation du spectre fréquentiel d'une image
- Analyse des composantes fréquentielles

### Autres Transformations
- **Histogramme** : Affiche la distribution des intensités de pixels
- **Égalisation d'histogramme** : Améliore le contraste global de l'image

## Installation

### Prérequis
- Python 3.x

### Dépendances
Installez les bibliothèques nécessaires avec la commande :

```bash
pip install streamlit opencv-python-headless pillow matplotlib numpy
```

## Utilisation

1. Lancez l'application avec la commande suivante :

```bash
streamlit run app.py
```

2. Votre navigateur web s'ouvrira automatiquement avec l'interface de l'application
3. Téléchargez une image depuis votre ordinateur
4. Naviguez entre les différentes sections pour appliquer les transformations souhaitées
5. Téléchargez l'image transformée si nécessaire

## Structure du Projet

```
├── app.py                  # Point d'entrée principal de l'application
├── utils/
│   ├── basic_transforms.py # Fonctions pour les transformations basiques
│   ├── advanced_transforms.py # Fonctions pour les transformations avancées
│   └── fourier_transform.py # Fonctions pour la transformation de Fourier
├── assets/                 # Images d'exemple et ressources
└── README.md               # Ce fichier
```

## Personnalisation

Vous pouvez facilement étendre l'application en :
- Ajoutant de nouvelles transformations dans les fichiers existants
- Créant de nouveaux modules pour des fonctionnalités supplémentaires
- Modifiant l'interface utilisateur dans app.py

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Commit vos changements (`git commit -m 'Add some amazing feature'`)
4. Push sur la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Auteur

Votre Nom

---

*Cette application a été créée dans un but éducatif pour démontrer les possibilités de traitement d'image avec Python et Streamlit.*