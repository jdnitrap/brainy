#!/bin/sh
# Downloads MNIST (about 11 MB compressed) into data/.
set -e
mkdir -p data
cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [ -f "$f" ] && continue
    curl -fsSL -o "$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
    gunzip -f "$f.gz"
done
ls -l
