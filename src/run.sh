#################### run

#!/bin/bash

dataset=$1
factors=(50)
lambda=(0.1 1)
alpha=( 0.0001 0.001)
s_user=(21 4)
s_item=(21 4)



for i in ${s_item[*]};
do	
	for u in ${s_user[*]};
	do	
		for f in ${factors[*]};
		do
			for l in ${lambda[*]};
			do
				for a in ${alpha[*]};
				do
					#echo "./main.bin 1 $dataset 5 0 13 4 $f $u $s_item 0 100 $l $a"
					./main.bin 1 $dataset 5 0 5 13 5 $f $u $i 0 100 $l $a
				done
			done
		done
	done
done
