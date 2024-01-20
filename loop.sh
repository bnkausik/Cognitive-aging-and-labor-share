#!/bin/sh

for i in {1..20} 
do
	python tf_m.py -v_flag 0 -plot_flag 0 -n_epochs 100 -lr 0.01 -fname "data/FI_klems.csv"  
done

