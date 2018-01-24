# Tuner

## Requrement 
- Python 2.7+ or Python3.5
- TensorFlow
- Keras
- Hypereras
- Augmentor

## Installation

```sh
pip install git+https://github.com/sktnkysh/tuner
```

## Usage 

```
Usage:
  format-dataset <dataset_dir> [options]

Options:
  --output                     classed datasset directory
  --teachin-file               csv or tsv file.
  --brain                      specific option
```

```
Usage:
  toon <dataset_dir> [options]
```

## Examples
### format-dataset
```sh
$ ls dataset-dog-cat/
cat1.jpg  cat2.jpg ...
dog1.jpg  dog2.jpg ...

$ format-dataset --brain dataset-dog-cat -o classed-dog-cat/
classed-dog-cat 

$ filecount classed-dog-cat/
./classed-dog-cat/cat 123
./classed-dog-cat/dog 133
./  256

$ format-dataset --brain dataset-dog-cat -o classed-dog-cat/ | toon -o best-model.hdf5 &
```

```sh
$ ls dataset-dog-cat/
label.csv 1.jpg  2.jpg  3.jpg 4.jpg ...

$ head label.csv
id,lable
1,dog
2,cat
3,cat
...

$ filecount format-dataset dataset-dog-cat -o classed-dog-cat/ --teaching-file label.csv
classed-dog-cat 

$  classed-dog-cat/
./classed-dog-cat/cat 123
./classed-dog-cat/dog 133
./  256

$ format-dataset dataset-dog-cat -o classed-dog-cat/ -t label.csv | toon -o best-model.hdf5 &
```
<a href='https://github.com/sktnkysh/filecount'>filecount</a>
