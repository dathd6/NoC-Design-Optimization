#!/bin/bash

TOURNAMENT_SIZE=200
POPULATION_SIZE=1000
ITERATIONS=100
EXPERIMENTS=2

source env/bin/activate

# test_case=("pip" "mpeg" "vopd" "G32" "G48" "G64" "G80" "G90" "G112" "G128")
# rows=(3, 4, 4, 6, 7, 8, 9, 10, 11, 12)
# columns=(3, 3, 4, 6, 7, 8, 9, 10, 11, 12)

test_case=("G48")
rows=(7)
columns=(7)

for index in "${!test_case[@]}"; do
  python3 optimize.py --core-graph "./dataset/bandwidth/${test_case[$index]}.txt" \
                      --rows ${rows[$index]} \
                      --columns ${columns[$index]} \
                      --experiments $EXPERIMENTS \
                      --population $POPULATION_SIZE \
                      --tournament $TOURNAMENT_SIZE \
                      --iterations $ITERATIONS \
                      --bi-level \
                      --nsga-ii

  python3 visualize.py --name ${test_case[$index]}
done 
