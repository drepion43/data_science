from typing import List
Vector=List[float]
def sum_of_squares(v:Vector)->float:
    return sum(v_i ** 2 for v_i in v)
from typing import Callable
def difference_quotient(f: Callable[[float],float],x:float,h:float)->float:
    return (f(x+h)-f(x))/h
def square(x:float)->float:
    return x*x

def derivative(x:float)->float:
    return 2*x

def partial_differene_quotient(f:Callable[[Vector],float],v:Vector,i:int,h:float)->float:
    w=[v_j + (h if j == i else 0) for j,v_j in enumerate(v)]
    return (f(w)-f(v))/h

def estimate_gradient(f:Callable[[Vector],float],v:Vector,h:float=0.0001):
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

import random
def add(v:Vector, w:Vector)->Vector:
    assert len(v)==len(w), "vectors must be the same length"
    
    return[v_i+w_i for v_i,w_i in zip(v,w)]
def sclar_multiply(c:float, v:Vector)->Vector:
    return[c*v_i for v_i in v]

def gradient_step(v: Vector,gradient:Vector,step_size:float)->Vector:
    step=sclar_multiply(step_size,gradient)
    return add(v,step)

def sum_of_squares_gradient(v:Vector)->Vector:
    return [2*v_i for v_i in v]

def linear_gradient(x:float,y:float,theta:Vector)->Vector:
    slope,intercept=theta
    predicted=slope*x+intercept # 모델의 예측값
    error=(predicted-y) # 오차값
    squared_error=error**2 # 오차제곱값
    grad=[2*error*x,2*error]
    return grad

def vector_mean(vectors:List[Vector])->Vector:
    n=len(vectors)
    return sclar_multiply(1/n,vector_sum(vectors))

def vector_sum(vectors:List[Vector])->Vector:
    assert vectors,"no vectors provided!"
    
    num_elements=len(vectors[0])
    assert all(len(v)==num_elements for v in vectors), "different sizes!"
    
    return [sum(vector[i] for vector in vectors) for i in range(num_elements) ]

from typing import TypeVar,List,Iterator

T=TypeVar('T')

def minibatches(dataset:List[T], batch_size:int,shuffle:bool=True)->Iterator[List[T]]:
    batch_starts=[start for start in range(0,len(dataset),batch_size)]
    if shuffle: random.shuffle(batch_starts)
    for start in batch_starts:
        end=start+batch_size
        yield dataset[start:end]