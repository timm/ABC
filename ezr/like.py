#!/usr/bin/env python3 -B
from data import *

#--------------------------------------------------------------------
def like(i:o, v:Any, prior=0) -> float :
  "probability of 'v' belong to the distribution in 'i'"
  if i.it is Sym:
    tmp = ((i.has.get(v,0) + the.m*prior) 
           /(sum(i.has.values())+the.m+1e-32))
  else:
    var = 2 * i.sd * i.sd + 1E-32
    z  = (v - i.mu) ** 2 / var
    tmp =  math.exp(-z) / (2 * math.pi * var) ** 0.5
  return min(1, max(0, tmp))

def likes(data:Data, row:Row, nall=100, nh=2) -> float:
  "How much does this DATA like row?"
  prior= (data.n + the.k) / (nall + the.k*nh)
  tmp= [like(c,v,prior) for c in data.cols.x if (v:=row[c.at]) != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)    

#--------------------------------------------------------------------
def eg__Sym(): 
  "Sym: demo of likelihood."
  print(s := round(like(adds("aaaabbc"), "a"),2))
  assert s == 0.44

def eg__num(): 
  "Num: demo."
  print(x := round(adds(random.gauss(10,2) for _ in range(1000)).sd,2))
  assert x == 2.05
  
#--------------------------------------------------------------------
def eg__all()             : mainAll(globals())
def eg__list()            : mainList(globals())
def eg_h()                : print(helpstring)
if __name__ == "__main__" : main(globals())
