# VisShop AI — Détection de vol en magasin par vision par ordinateur

Implémentation du *Cahier de Challenge VisShop AI* : analyse temps réel des flux de caméras 2D existantes,
détection de **4 comportements de vol**, alerte visuelle (cadre rouge clignotant) et journalisation horodatée.

| # | Comportement | Seuil |
|---|---|---|
| 1 | Vol en poche | 0.72 |
| 2 | Caché sous vêtement | 0.78 |
| 3 | Caché dans sac à dos | 0.70 |
| 4 | Caché dans sac à main | 0.75 |

## Pipeline (6 étapes)

```
OpenCV capture → YOLOv11n (YOLOv8n fallback CPU) → DeepSORT → MediaPipe Holistic (1 frame / N)
→ Moteur de scoring (4 comportements) → Filtre anti-fausses alertes → AlertManager + Overlay
```

| Étape | Module | Fréquence |
|---|---|---|
| 1 Capture | `visshop/video_source.py` | chaque frame |
| 2 Détection personnes (+ sacs) | `visshop/yolo_detector.py`, `model_selector.py` | chaque frame |
| 3 Suivi + ReID | `visshop/deepsort_tracker.py` | chaque frame |
| 4 Pose | `visshop/pose_estimator.py` | 1 frame / N (`frame_skipper.py`) |
| 5 Classification | `visshop/action_classifier.py`, `features.py` | 1 frame / N |
| 5b Filtre 4 niveaux | `visshop/false_alert_filter.py` | 1 frame / N |
| 6 Alertes + overlay | `visshop/alert_manager.py`, `overlay.py` | chaque frame |

Orchestration : `visshop/pipeline.py` (tous les modules sont injectés → remplaçables indépendamment).

### Scoring (par comportement, meilleur des deux poignets)
`score = 0.30·proximité + 0.25·caché + 0.15·approche + 0.15·préhension + 0.15·visite_rayon`
- **caché** : *visibilité* du poignet MediaPipe (< 0.25 ⇒ main cachée) — remplace l'axe Z instable en 2D ;
- **proximité** : poignet près de la poche / du buste / de la bbox du sac associé à la personne ;
- **approche** : le poignet s'est rapproché de la cible sur ~12 échantillons ;
- **préhension** : main fermée (objet tenu) dans les 3 dernières secondes ;
- **visite_rayon** : passage récent (≥ 1 s) dans la zone rayon.

### Filtre anti-fausses alertes (4 niveaux)
1. **Règles contextuelles** : visite du rayon (durée ≥ 1 s, validité 8 s) ; cabine d'essayage ⇒ détection désactivée ; personnel (uniforme HSV) ignoré.
2. **Lissage** : moyenne pondérée (poids croissants) sur 20 échantillons analysés.
3. **Hystérésis** : déclenchement si lissé ≥ 0.68 **et** score brut ≥ seuil du comportement ; annulation sous 0.35 ; ≥ 8 échantillons requis.
4. **Cooldown** : une ligne de journal max par personne toutes les 3 s.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install -r requirements.txt                         # production (YOLO, DeepSORT, MediaPipe)
# ou, pour la démo + tests uniquement (léger, sans GPU) :
pip install -r requirements-test.txt
```

## Utilisation

```bash
# Démo sans caméra ni modèle (scénarios synthétiques)
python main.py --demo list
python main.py --demo theft_handbag --save-video demo.mp4
python main.py --demo theft_pocket --display           # q pour quitter

# Flux réel
python main.py --source 0 --display                                    # webcam USB
python main.py --source rtsp://user:pwd@192.168.1.10/stream --config configs/config.example.json
python main.py --source video.mp4 --device cpu --save-video out.mp4
```

Sorties dans `output/` : `alerts.jsonl` (un JSON par alerte : horodatage ISO, id, comportement, score, bbox,
capture) et `screenshots/*.jpg`.

**Configuration** : `configs/config.example.json` — seuils, filtre, **zones rayon / cabine** (à dessiner pour chaque
caméra, coordonnées normalisées), plages HSV de l'uniforme, frame skipping.

## Rendu vidéo « AI SECURITY MONITOR »

Par défaut (`--style monitor`) la vidéo annotée reprend le style d'une console de sécurité :
bandeau `AI SECURITY MONITOR  mm:ss | Held Items: n | Alerts: k` (ligne verte, rouge pendant une alerte),
étiquette verte `ID: n` par personne, squelette des bras (vert, rouge si suspect), cadre rouge
`SUSPECT n` + repère `Item ?` près de la main cachée, panneau **INCIDENT RECORD** (heure, ID, comportement,
vignette de preuve) pendant 6 s, puis écran final **INCIDENT REPORT SUMMARY** (3 s).
`--style classic` = cadre rouge clignotant simple ; `--no-summary` supprime l'écran final.

## Utiliser une vraie vidéo (ex. `sac.mp4`)

```bash
pip install -r requirements.txt
python main.py --source sac.mp4 --config configs/real_video.json --device cpu \
       --save-video resultat.mp4 --timeline-csv scores.csv
```
- `resultat.mp4` : vidéo annotée (cadre vert = personne, **cadre rouge clignotant = alerte**).
- `scores.csv` : scores bruts et lissés de chaque comportement dans le temps (pour régler les seuils).
- `configs/real_video.json` : sans zones rayon/cabine (caméra mobile ou gros plan). Pour une caméra fixe de
  magasin, dessinez les zones (voir `config.example.json`) : c'est la meilleure protection contre les fausses alertes.

Adaptations faites pour les vraies vidéos : géométrie robuste aux **gros plans** (hanches hors cadre
ignorées), critère « préhension » retiré et poids renormalisés quand les **mains ne sont pas détectées**,
**mémoire des sacs** (1.5 s) quand YOLO rate le sac, tolérance aux pertes brèves de pose.

## Matériel
- **GPU (production)** : NVIDIA GTX 1660 6 Go min → YOLOv11n, MediaPipe 1 frame / 3.
- **CPU seul** : bascule auto sur YOLOv8n + Frame Skipping ×5 (adaptatif) ; performances réduites.
- Forcer : `--device cpu|cuda` ou variable `VISSHOP_DEVICE`.

## Tests

```bash
pip install -r requirements-test.txt
pytest                                  # ~70 tests, ~30 s, sans GPU ni modèle
python scripts/evaluate_synthetic.py    # rapport précision / rappel / F1 / FP-h / latence
```

Les tests couvrent : géométrie, config, sélection modèle/frame skipping, features, scoring des 4 comportements,
chaque niveau du filtre, trackers, parsing YOLO et conversion MediaPipe (faux modèles), alertes JSONL/captures,
overlay clignotant, et **10 scénarios de bout en bout** (4 vols, comportements normaux, cabine, employé, geste rapide,
deux clients) avec vérification des cibles du cahier (précision ≥ 92 %, rappel ≥ 87 %, FP/h < 2, latence < 2.5 s).

## ⚠️ Limites à connaître (honnêteté)
- Les métriques ci-dessus sont obtenues sur des **scénarios synthétiques** : elles valident la logique, **pas** la
  performance réelle. Les chiffres du cahier (92 % / 87 %…) devront être mesurés sur des **vidéos réelles annotées**
  du site pilote ; les poids/seuils de scoring se règlent alors sur ces données.
- `yolo_detector`, `deepsort_tracker` (DeepSORT) et `pose_estimator` (MediaPipe) sont testés avec de faux
  backends : ils n'ont **pas** été exécutés avec les vrais modèles dans cet environnement (installer `requirements.txt`
  et lancer sur une vidéo réelle pour valider).
- Pas de détecteur d'objet « produit » : l'objet en main est inféré (préhension + passage au rayon).
- Le sac à dos porté dans le dos est difficilement visible en caméra 2D frontale.
- Le suivi sans DeepSORT (repli `SimpleIoUTracker`) n'a pas de ReID visuelle.
