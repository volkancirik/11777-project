#!/usr/bin/env/sh

device=$1
folder=$2
mtype=$3

for m in `find $folder -name \*.meta | cut -d "." -f1-2`
do
    echo "THEANO_FLAGS=device=$device python test_translator.py --path $m --mtype $mtype > $m.output"
    THEANO_FLAGS=device=$device python test_translator.py --path $m --mtype $mtype > $m.output
done
