#!/bin/sh

DATAURL="https://archive.ics.uci.edu/ml/machine-learning-databases/00288/leaf.zip"

if hash aria2c 2>/dev/null; then
    DLER="$(which aria2c)"
elif hash wget 2>/dev/null; then
    DLER="$(which wget)"
else
    DLER="$(which curl)"
fi

$DLER $DATAURL

unzip leaf.zip

rm -R -f ./BW ./RGB ./leaf.zip
