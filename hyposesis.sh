#!/bin/sh

for i in `seq 1 3000`
do
  x=$(printf "%.2f" $(echo "scale=2; 0.1 * $i" | bc ))
  python3 scripts/hypothesis_testing.py -r $x -g data_labels/bbox/ead2019_000/ground-truth
done 
