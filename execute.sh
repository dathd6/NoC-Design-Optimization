#!/bin/bash

TOURNAMENT_SIZE=200
POPULATION_SIZE=1000
ITERATIONS=1000
EXPERIMENTS=9

virtualenv env
source env/bin/activate
pip install -r requirements.txt

test_case=("G1024") # ("pip" "mpeg" "vopd" "auto_industry" "G32" "G48") # "G64" "G80" "G90" "G112" "G128")
rows=(32) #, 8, 9, 10, 11, 12)
columns=(32) #, 8, 9, 10, 11, 12)
# colors=('#EA1000' '#129012' '#E19124' '#A39181' '#3adfff' '#F2E329')
colors=('#F2E329')

for index in "${!test_case[@]}"; do
  python3 optimize.py --core-graph "./dataset/bandwidth/${test_case[$index]}.txt" \
                      --rows ${rows[$index]} \
                      --columns ${columns[$index]} \
                      --experiments $EXPERIMENTS \
                      --population $POPULATION_SIZE \
                      --tournament $TOURNAMENT_SIZE \
                      --iterations $ITERATIONS \
                      --nsga-ii \
                      --bi-level
                      # --old-population \

  python3 visualize.py --application ${test_case[$index]} \
                       --color ${colors[$index]} \
                       --objective-space \
                       --optimal-fitness \
                       --algorithm-animation \
                       --convergence-plot
done 
