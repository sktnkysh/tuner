# Tuner

## Requrement 
- Python 2.7+ or Python3.5
- Keras
- Hypereras
- Augmentor

## Installation

```
python cmdline-flow-eyes.py -d micin-dataset/eyes -t label.tsv
python cmdline-flow-brain.py -d micin-datatset/brain
```

```
python format_dataset.py ../micin-dataset/brain -o tmp/BRAIN --brain
python format_dataset.py ../micin-dataset/eyes -o EYES -t label.tsv
```

```
./format_dataset.py ../micin-dataset/brain --brain | ./toon.py
```
