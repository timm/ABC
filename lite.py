#!/usr/bin/env python3 -B 
from types import SimpleNamespace as o
import random, bisect, math, sys, re

pos = bisect.bisect_left

the=o(acq="klass", 
      Any=4,      
      Build=24, 
      Check=5,  
      Few=128,
      bins=7,  
      k=1,
      m=2,
      p=2,
      seed=1234567891,
      file="../moot/optimize/misc/auto93.csv")

#--------------------------------------------------------------------
def adds(it,src): [add(it,x) for x in src]; return it

def Data(src):
  src  = iter(src)
  data = o(ok=0, rows=[], names=next(src))
  data.cols = [[] for _ in data.names]
  return adds(data, src)

def clone(data, rows=None):
  return adds(Data([data.names]), rows or [])

def add(data, row):
  [c.append(x) for c,x in zip(data.cols,row) if x != "?"]
  data.rows += [row]
  data.ok = False
  return row

def sub(data, row, zap=False):
  print([c for c,x in zip(data.cols,row) if x != "?"])
  [c.pop(pos(c,x)) for c,x in zip(data.cols,row) if x != "?"]
  if zap: data.rows.remove(row)
  return row

def per(data,c,x): 
  c=ok(data).cols[c]; return pos(c,x) / len(c) 

def bin(data,c,x): 
  b=int(per(data,c,x)*the.bins); return min(b,the.bins-1)

def norm(data,c,x): 
  lo,*_,hi = ok(data).cols[c]; return (x-lo)/(hi-lo+1e-32)

def ok(data):
  if not data.ok: [col.sort() for col in data.cols]
  data.ok=True
  return data

#--------------------------------------------------------------------
def atom(s):
  for fn in [int,float]:
    try: return fn(s)
    except Exception as _: pass
  s = s.strip()
  return {'True':True,'False':False}.get(s,s)

def csv(file):
  with open(file,encoding="utf-8") as f:
    for line in f:
      if (line := line.split("%")[0]):
        yield [atom(s.strip()) for s in line.split(",")]

#--------------------------------------------------------------------
def eg__data(): 
  print(ok(Data(csv(the.file))).cols[3])

def eg__per():
  data = Data(csv(the.file))
  print(per(data,3,82))
  print(bin(data,3,82))
  print(norm(data,3,82))

def eg__inc():
  d1 = Data(csv(the.file))
  d2 = clone(d1)
  x  = d2.cols[1]
  print(1)
  for row in d1.rows:
    add(d2,row)
    if len(d2.rows)==100:  
      mid1 = x[len(x)//2]; print("asIs",mid1)
  for row in d1.rows[::-1]:
    if len(d2.rows)==100: 
      mid2 = x[len(x)//2]; print("asIs",mid2)
      #assert abs(mu2 - mu1) < 1.01 and abs(sd2 - sd1) < 1.01
    sub(d2,row,zap=True)

#--------------------------------------------------------------------
def main(funs):
  for n,arg in enumerate(sys.argv):
    if (fn := funs.get(f"eg{arg.replace('-', '_')}")):
      random.seed(the.seed); fn()
    else:
      for key in vars(the):
        if arg=="-"+key[0]: the.__dict__[key]=atom(sys.argv[n+1])

#--------------------------------------------------------------------
if __name__ == "__main__": main(globals())
