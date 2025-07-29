#!/usr/bin/env python3 -B 
"""
lite.py, lightweight multi objective.
(col) 2025, Tim Menzies <timm@ieee.org>, MIT license

 -A  Any=4           on init, how many initial guesses?
 -B  Build=24        when growing theory, how many labels?
 -C  Check=5         when testing, how many checks?
 -F  Few=128
 -b  bins=7          number of bins
 -k  k=1
 -m  m=2
 -p  p=2
 -s  seed=1234567891 random number seed
 -f  file=../moot/optimize/misc/auto93.csv
                     path to CSV file

 -h                  show help
 --all               run all examples.
"""
from types import SimpleNamespace as o
import random, math, sys, re

def atom(s):
  for fn in [int,float]:
    try: return fn(s)
    except Exception as _: pass
  s = s.strip()
  return {'True':True,'False':False}.get(s,s)

the = o(**{k:atom(v) for k,v in re.findall(r"(\w+)=(\S+)",__doc__)})

#--------------------------------------------------------------------
def Sym(at=0, txt=""): 
  return o(it=Sym, at=at,txt=txt,has={})

def Num(at=0, txt=" "): 
  return o(it=Num, at=at, txt=txt, lo=1e32, mu=0, m2=0, sd=0, n=0, 
           hi=-1e32, more = 0 if txt[-1] == "-" else 1)

def Data(src):
  src = iter(src)
  return adds(src, o(it=Data, n=0, rows=[], cols= Cols(next(src))))

def Cols(names):
  all, x, y, klass = [],[],[],None
  for c,s in enumerate(names):
    all += [(Num if s[0].isupper() else Sym)(c,s)]
    if s[-1] == "X": continue
    if s[-1] == "!": klass = all[-1]
    (y if s[-1] in "!-+" else x).append(all[-1])
  return o(it=Cols, names=names, all=all, x=x, y=y, klass=klass)

def clone(data, rows=None):
  return adds(rows or [], Data([data.cols.names]))

#--------------------------------------------------------------------
def adds(src, it=None):
  it = it or Num()
  [add(it, x) for x in src]
  return it

def sub(x, v, zap=False): return add(x,v,-1,zap)

def add(x, v, inc=1, zap=False):
  if v == "?": return v
  if x.it is Sym: x.has[v] = inc + x.has.get(v,0)
  elif x.it is Num:
    x.n += inc
    x.lo, x.hi = min(v, x.lo), max(v, x.hi)
    if inc < 0 and x.n < 2:
      x.sd = x.m2 = x.mu = x.n = 0
    else:
      d     = v - x.mu
      x.mu += inc * (d / x.n)
      x.m2 += inc * (d * (v - x.mu))
      x.sd  = 0 if x.n < 2 else (max(0,x.m2)/(x.n-1))**.5
  elif x.it is Data:
    x.n += inc
    if inc > 0: x.rows += [v]
    elif zap: x.rows.remove(v) # slow for long rows
    [add(col, v[col.at],inc) for col in x.cols.all]
  return v

#--------------------------------------------------------------------
def dist(src):
  d,n = 0,0
  for v in src: n,d = n+1, d + v**the.p
  return (d/n) ** (1/the.p)

def disty(data, row):
  return dist(abs(norm(c, row[c.at]) - c.more) for c in data.cols.y)

def distysort(data,rows=None):
  return sorted(rows or data.rows, key=lambda r: disty(data,r))

def norm(i, v): 
  return v if v=="?" or i.it is Sym else (v-i.lo)/(i.hi-i.lo + 1E-32)

#--------------------------------------------------------------------
def csv(file):
  with open(file,encoding="utf-8") as f:
    for line in f:
      if (line := line.split("%")[0]):
        yield [atom(s.strip()) for s in line.split(",")]

def shuffle(rows):
  random.shuffle(rows)
  return rows

def pout(x): print(out(x))

def out(x):
  if callable(x): x= x.__name__
  if type(x) is float: x = int(x) if int(x)==float(x) else f"{x:.3f}"
  if hasattr(x,"__dict__"):
    x= "{" + ' '.join(f":{k} {out(v)}" 
             for k,v in x.__dict__.items() if str(k)[0] != "_") + "}"
  return str(x)

#---------------------------------------------------------------------
def like(i, v, prior=0):
  if i.it is Sym:
    tmp = ((i.has.get(v,0) + the.m*prior)
           /(sum(i.has.values())+the.m+1e-32))
  else:
    var = 2 * i.sd * i.sd + 1E-32
    z  = (v - i.mu) ** 2 / var
    tmp =  math.exp(-z) / (2 * math.pi * var) ** 0.5
  return min(1, max(0, tmp))

def likes(data, row, nall=100, nh=2):
  "How much does this DATA like row?"
  prior = (len(data.rows) + the.k) / (nall + the.k*nh)
  tmp = [like(col,v,prior) 
         for col in data.cols.x if (v:=row[col.at]) != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)    

def likely(data,rows=None):
  rows = rows or data.rows
  no   = clone(data, shuffle(rows[:]))
  yes  = clone(data)
  while no.n > 2 and yes.n < the.Build:
    if yes.n <= the.Any: 
      add(yes, sub(no, no.rows.pop()))
    if yes.n == the.Any:
      yes.rows = distysort(yes)
      n    = round(the.Any**.5)
      best = clone(data, yes.rows[:n])
      rest = clone(data, yes.rows[n:])
    if yes.n > the.Any:
      add(yes, add(best, likely1(best, rest, no)))
      if best.n > yes.n**.5:
        best.rows = distysort(yes,best.rows)
        while best.n > yes.n**.5:
          add(rest, sub(best, best.rows.pop(-1)))
  return distysort(yes)
 
def likely1(best,rest,no):
  shuffle(no.rows)
  j, nall = 0, best.n + rest.n
  for i,row in enumerate(no.rows[:the.Few*2]):
    if likes(best,row,2,nall) > likes(rest,row,2,nall):
      j = i; break
  return sub(no, no.rows.pop(j))

#--------------------------------------------------------------------
def eg_h(): 
  print(__doc__)

def eg__the(): 
  print(the)

def eg__sym(): 
  pout(adds("aaaabbc",Sym()))

def eg__Sym(): 
  s = adds("aaaabbc",Sym()); assert 0.44 == round(like(s,"a"),2)

def eg__num(): 
  pout(adds(random.gauss(10,2) for _ in range(1000)))

def eg__Num() : 
  n = adds(random.gauss(10,2) for _ in range(1000))
  assert 0.13 == round(like(n,10.5),2)

def eg__data():
  [pout(x) for x in Data(csv(the.file)).cols.y]

def eg__bayes():
  data = Data(csv(the.file))
  assert all(-30 <= likes(data,t) <= 0 for t in data.rows)
  print(sorted([round(likes(data,t),2) for t in data.rows])[::20])

def eg__inc():
  d1 = Data(csv(the.file))
  d2 = clone(d1)
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

def eg__likely():
  data = Data(csv(the.file))
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda n: int(100*(1 - (n - b4.lo) / (b4.mu - b4.lo)))
  now  = adds(disty(data, likely(data)[0]) for _ in range(2))
  print(out(win(now.mu)), re.sub(r".*/","",the.file))

def eg__all(): 
 for s,fn in globals().items():
   if s != "eg__all" and s.startswith("eg_"): 
     print(f"\n--| {s} |--------------"); random.seed(the.seed); fn()

#--------------------------------------------------------------------
if __name__ == "__main__":
  for n,arg in enumerate(sys.argv):
    if (fn := globals().get(f"eg{arg.replace('-', '_')}")):
      random.seed(the.seed); fn()
    else:
      for key in vars(the):
        if arg=="-"+key[0]: the.__dict__[key] = atom(sys.argv[n+1])
