# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/sh
mkdir data
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$1.zip -P data
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$2.zip -P data
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$1-$2.5000-6500.txt -P data
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$2-$1.5000-6500.txt -P data
