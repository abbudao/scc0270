import numpy as np
import os
class perceptron(object):

    """Classe representando um Perceptron. """

    def __init__(self,dir,eta=0.01):
        """Inicia os valores e calcula o peso para training set contido em dir """
        self.loop=0
        self.bias=0
        self.eta=eta
        self.weights=np.ones((5,5),dtype=np.int)
        self.xmatrix,self.expected=self.read_samples(dir)
        self.train()


    def read_samples(self,dir):
        """Le todos os arquivos contidos na pasta dir, e armazena os valores esperados em xmatrix e sua estrututra de dados em xmatrix"""
        expected=[]
        xmatrix=[]
        for root,dirs,files in os.walk(dir):
            for file in files:
                with open(os.path.join(root,file),"r") as auto:
                    expected.append(int(auto.readline().strip('\n')))
                    a=[]
                    for line in auto:
                        a.append([int(n) for n in line.strip('\n').split(' ')])
                    xmatrix.append(a)
        return np.asarray(xmatrix),expected

    def train(self):
        """Treina o perceptron"""
        stout=[]
        while self.expected!=stout:
            self.loop=self.loop+1
            stout=[]
            for matriz in self.xmatrix:
                stout.append(np.vdot(matriz,self.weights)+self.bias)
                stout= [1 if a>0   else -1 for a in stout ]
                for i,calculated in enumerate(stout):
                    if(calculated!=self.expected[i]):
                        erro=self.expected[i]-calculated
                        self.bias=erro*self.eta+self.bias
                        for j,weight_line in enumerate(self.weights):
                            for k,weight in enumerate(weight_line):
                                self.weights[j][k]=weight+erro*self.eta*self.xmatrix[i][j][k]


    def test(self,dir):
        """Testa o perceptron com os arquivos contidos na pasta dir"""
        matrix_train,expected_train= self.read_samples(dir)
        stout=[]
        for matriz in matrix_train:
            stout.append(np.vdot(matriz,self.weights)+self.bias)
            stout= [1 if a>0   else -1 for a in stout ]

        print("Resultado esperado do teste:")
        print(expected_train)
        print("Resultado obtido no teste:")
        print(stout)
        return 

    def print_stats(self):
        """Mostra na tela informações relevantes do objeto """
        print("==== Informações deste Perpectron ==== ")
        print("Bias:")
        print(self.bias)
        print("Pesos:")
        print(self.weights)
        print("Loops até convergência:")
        print(self.loop)



#Teste sequencial de funcionamento
percep=perceptron("sample")
percep.test("test")
percep.print_stats()
