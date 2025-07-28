#!/usr/bin/env python3 -B
import sys,re
sys.path += ["../src"]
from abc101 import *

def all_egs(run=False):
  "Run all eg__* functions."
  for k,fn in globals().items():
    if k.startswith('eg__') and k != 'eg__all':
      if run:
        print("\n----["+k+"]"+'-'*40)
        random.seed(the.seed)
        fn()
      else:  
        print("  "+re.sub('eg__','--',k).ljust(10),"\t",fn.__doc__ or "")

#--------------------------------------------------------------
def daBest(data,rows=None):
  rows = rows or data.rows
  Y=lambda r: disty(data,r)
  return Y(sorted(rows, key=Y)[0])

def eg_h(): 
  "Print help."
  print(__doc__)

def eg__all():
  "Run all egs."
  all_egs(run=True)

def eg__list():
  "List all egs."
  print("\n./abc.py [OPTIONS]\n")
  all_egs()

def eg__the(): 
  "Show the config."
  print(the)

def eg__sym(): 
  "Demo  of Syms."
  print(adds("aaaabbc",Sym()))

def eg__Sym(): 
  "Another demo of Syms."
  s=adds("aaaabbc",Sym()); assert 0.44 == round(like(s,"a"),2)

def eg__Num() : 
  "Demo of Nums."
  n = adds(random.gauss(10,2) for _ in range(1000))
  assert 0.14 == round(like(n,10.5),2)

def eg__Data(): 
  "Demo of Datas."
  [print(c) for c in Data(csv(the.file)).cols.all]

def eg__inc():
  "Check we can add/delete rows incrementally."
  d1 = Data(csv(the.file))
  d2 = dataClone(d1)
  x  = d2.cols.x[1]
  for row in d1.rows:
    add(d2,row)
    if len(d2.rows)==100:  
      mu1,sd1 = x.mu,x.sd 
      print(mu1,sd1)
  for row in d1.rows[::-1]:
    if len(d2.rows)==100: 
      mu2,sd2 = x.mu,x.sd
      print(mu2,sd2)
      assert abs(mu2 - mu1) < 1.01 and abs(sd2 - sd1) < 1.01
      break
    sub(d2,row,zap=True)

def eg__bayes():
  "Calculate Bayesian likelihoods."
  data = Data(csv(the.file))
  assert all(-30 <= likes(data,t) <= 0 for t in data.rows)
  print(sorted([round(likes(data,t),2) for t in data.rows])[::20])

def daRx(t):
    if not isinstance(t,(tuple,list)): return str(t)
    return ':'.join(str(x) for x in t)

def eg__disty():
  "Calculate distance to 'most'."
  data = Data(csv(the.file))
  data.rows.sort(key=lambda row: disty(data,row))
  assert all(0 <= disty(data,r) <= 1 for r in data.rows)
  print(', '.join(data.cols.names))
  print("top4:");   [print("\t",row) for row in data.rows[:4]]
  print("worst4:"); [print("\t",row) for row in data.rows[-4:]]

def eg__tree():
  treeShow(Tree(Data(csv(the.file))))

def eg__likely():
  data = Data(csv(the.file))
  treeShow( Tree(dataClone(data, likely(data,data.rows).labels)))

#-----------------------------------------------------------
if __name__ == "__main__":
  for n,arg in enumerate(sys.argv):
    if (fn := globals().get(f"eg{arg.replace('-', '_')}")):
      random.seed(the.seed)
      fn() 
    else:
      for key in vars(the):
        if arg == "-"+key[0]: 
          the.__dict__[key] = coerce(sys.argv[n+1])
