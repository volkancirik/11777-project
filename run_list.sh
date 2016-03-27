#!/usr/bin/env/sh

device=$1
list=$2
mtype=$3
prefix=$4

for m in `echo $list | awk '{for(i=1;i<=NF;i++)print $i}'`
do
    echo "THEANO_FLAGS=device=$device python test_translator.py --path $prefix/$m --mtype $mtype > $prefix/$m.output"
    THEANO_FLAGS=device=$device python test_translator.py --path $prefix/$m --mtype $mtype > $prefix/$m.output

done
