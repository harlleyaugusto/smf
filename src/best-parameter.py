### SELECIONA MELHOR CONFIGURACAO E COPIA OS ARQUIVOS  .out .breakdown PARA PASTA final
### EXECUTAR NA PASTA EM QUE OS ARQUIVOS .OUT ESTAO
### PRECISA CRIAR PASTA CHAMADA 'final' 

import pandas as pd
import numpy as np
from scipy import stats

import shutil
import os
import glob

max_value = -1
max_file = ""

for f in glob.glob('./*.out'):
    df = pd.read_csv(f, header=None)
    value =  np.mean(df.ix[:,0])
    if(value > max_value):
        max_value = value
        max_file = f


"{0} {1}".format(max_file, max_value)
print(max_file)        

s_user = [4, 21]
s_item = [4, 21]


factors = [50]
regularization = [0.1, 1]
alpha = [0.0001, 0.001]

for u in s_user:
        for i in s_item:
                max = -1
                max_line = ""
                max_file = ""
                max_pair = ""
                for f in factors:
                        for r in regularization:
                                for a in alpha:
                                        s = "{0}-{1}-{2}-0-{3:.5f}-{4:.5f}".format(f,u,i,r,a)
                                        df = pd.read_csv(s+".out", header=None)
                                        value =  np.mean(df.ix[:,0])
                                        #print(value)
                                        line = "{0},{1},{2},0,{3},{4},{5}".format(u,i,f,r,a,value)
                                        if(value > max):
                                            max = value
                                            max_line = line
                                            max_file = s
                                            max_pair = "{0}-{1}".format(u,i)
		print(max_line)
