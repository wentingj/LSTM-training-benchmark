#!/bin/sh
lscpu
#rm LSTM
#icc LSTM.c  -fno-alias -restrict -qopt-prefetch=3 -xCOMMON-AVX512 -qopt-report=5 -qopenmp -lpthread -lm -lmkl_rt -ldl -O3 -g -o LSTM
icpc LSTM_opt3.c  -fno-alias -restrict -qopt-prefetch=3 -qopt-report=5 -qopenmp -lpthread -lm -lmkl_rt -ldl -O3 -march=native -g -o LSTM_opt3
#export KMP_BLOCKTIME=1
export KMP_AFFINITY=verbose,granularity=core,noduplicates,compact,0,0
#bdw
#export OMP_NUM_THREADS=44
#skx6148
#export OMP_NUM_THREADS=40
#knl
#export OMP_NUM_THREADS=68
#skx8180
export OMP_NUM_THREADS=56
#knm
#export OMP_NUM_THREADS=72

#./LSTM_opt2  --loops=100 --batch_size=64 --time_step=10 --input_dim=150 --hid=1024 
./LSTM_opt3  --loops=100 --batch_size=32 --time_step=30 --input_dim=128 --hid=128 
#./LSTM_opt2  --loops=100 --batch_size=32 --time_step=30 --input_dim=512 --hid=512 
#./LSTM_opt2  --loops=100 --batch_size=32 --time_step=30 --input_dim=1024 --hid=1024 
#./LSTM_opt2  --loops=100 --batch_size=128 --time_step=30 --input_dim=128 --hid=128 
#./LSTM_opt2  --loops=100 --batch_size=128 --time_step=30 --input_dim=512 --hid=512
#./LSTM_opt2  --loops=100 --batch_size=128 --time_step=30 --input_dim=1024 --hid=1024 
