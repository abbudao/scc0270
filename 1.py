import numpy as np
def sigmoid(x):
    """TODO: Docstring for sigmoid.

    :num: TODO
    :returns: TODO

    """
    return (1/(1+np.exp(-x)))

def output(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,input1,input2):
    """TODO: Docstring for output.
    :returns: TODO
    """
    return(sigmoid(teta3+w13*sigmoid(teta1+w11*input1+w12*input2)+w23*sigmoid(teta2+input1*w21+input2*w22)))

def weight_updates(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,input1,input2,expected):
    outputn=output(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,input1,input2)
    error=expected-outputn
    w13n=w13-0.5*sigmoid(teta1+w11*input1+w12*input2)*error
    w23n=w23-0.5*sigmoid(teta2+w21*input1+w22*input2)*error
    w11n=w11-0.5*error*input1
    w12n=w12-0.5*error*input2
    w21n=w21-0.5*error*input1
    w22n=w22-0.5*error*input2
    teta1n=teta1-0.5*error*sigmoid(1)
    teta2n=teta2-0.5*error*sigmoid(1)
    teta3n=teta3-0.5*error*sigmoid(1)
    print("Output:"+str(outputn))
    print("Error:"+str(error))
    print("W11:"+str(w11n))
    print("W12:"+str(w12n))
    print("W21:"+str(w21n))
    print("W22:"+str(w22n))
    print("W13:"+str(w13n))
    print("W23:"+str(w23n))
    print("teta1:"+str(teta1n))
    print("teta2:"+str(teta2n))
    print("teta3:"+str(teta3n))
    return(w11n,w12n,w21n,w22n,w13n,w23n,teta1n,teta2n,teta3n)


w11,w12,w21,w22,w13,w23,teta1,teta2,teta3=weight_updates(0.4,0.5,0.8,0.8,-0.4,0.9,-0.6,-0.2,-0.3,0,0,0)
print()
print("Iteracao 2")
w11,w12,w21,w22,w13,w23,teta1,teta2,teta3=weight_updates(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,0,1,1)
print()
print("Iteracao 3")
w11,w12,w21,w22,w13,w23,teta1,teta2,teta3=weight_updates(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,1,0,1)
print()
print("Iteracao 4")
w11,w12,w21,w22,w13,w23,teta1,teta2,teta3=weight_updates(w11,w12,w21,w22,w13,w23,teta1,teta2,teta3,1,1,0)
