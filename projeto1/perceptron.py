import numpy as np
import os
class perceptron(object):

    """Docstring for perceptron. """

    def __init__(self):
        """TODO: to be defined1. """
        
#Reading files
def read_samples(dir):
    print("====read_samples output======")
    matrix=[]
    expected=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            print("Arquivo lido:"+file)
            with open(os.path.join(root,file),"r") as auto:
                expected.append(int(auto.readline().strip('\n')))
                a=[]
                for line in auto:
                    a.append([int(n) for n in line.strip('\n').split(' ')])
                matrix.append(a)
    return np.asarray(matrix),expected

def training(x_matrix,results):
    print("====training output======")
    bias=0
    loop=0
    weights=np.ones((5,5),dtype=np.int)
    stout=[]
    while results!=stout:
        loop=loop+1
        stout=[]
        eta=0.01
        for matriz in x_matrix:
            stout.append(np.vdot(matriz,weights)+bias)
            stout= [1 if a>0   else -1 for a in stout ]
            for i,calculated in enumerate(stout):
                if(calculated!=results[i]):
                    erro=results[i]-calculated
                    bias=erro*eta+bias
                    for j,weight_line in enumerate(weights):
                        for k,weight in enumerate(weight_line):
                            weights[j][k]=weight+erro*eta*x_matrix[i][j][k]
    print("Esperado:")
    print(results)
    print("Obtido:")
    print(stout)
    print("Bias:")
    print(bias)
    print("Pesos:")
    print(weights)
    print("Iterações até convergência:")
    print(loop)
    return weights,bias

def test(weights,bias,dir):
    matrix,expected= read_samples(dir)
    print("====test output======")
    stout=[]
    for matriz in matrix:
        stout.append(np.vdot(matriz,weights)+bias)
        stout= [1 if a>0   else -1 for a in stout ]
    print("Esperado:")
    print(expected)
    print("Obtido:")
    print(stout)




#Atualizando o peso
matriz,esperado=read_samples('sample')
weights,bias=training(matriz,esperado)
test(weights,bias,'test')
