rm *.out

make

SIZE=(4000 2000 1000 500)
THREADS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for s in ${SIZE[@]}; do
   for t in ${THREADS[@]}; do
	qsub -v SIZE=$s,THREADS=$t jacobi.job
   done
done

qstat -u as15656
