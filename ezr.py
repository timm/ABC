#!/usr/bin/env python3 -B 
"""
lite.py, lightweight multi objective.   
(col) 2025, Tim Menzies <timm@ieee.org>, MIT license   
   
     -a  acq=klass       acquisition function   
     -A  Any=4           on init, how many initial guesses?   
     -B  Build=24        when growing theory, how many labels?   
     -C  Check=5         when testing, how many checks?   
     -F  Few=128         sample size of data random sampling  
     -b  bins=7          number of bins   
     -k  k=1             bayes low frequency hack  
     -m  m=2             bayes low frequency hack  
     -p  p=2             distance co-effecient
     -s  seed=1234567891 random number seed   
     -f  file=../moot/optimize/misc/auto93.csv  data file 
       
     -h                  show help   
     --all               run all examples.   
"""
from types import SimpleNamespace as o
import random, math, sys, re

def atom(s:str) -> int|float|str|bool:
  "string coerce"
  for fn in [int,float]:
    try: return fn(s)
    except Exception as _: pass
  s = s.strip()
  return {'True':True,'False':False}.get(s,s)

Row=list[atom]
the = o(**{k:atom(v) for k,v in re.findall(r"(\w+)=(\S+)",__doc__)})

#--------------------------------------------------------------------
def Sym(at=0, txt="") -> o: 
  "summarize symbols"
  return o(it=Sym, at=at,txt=txt,has={})

def Num(at=0, txt=" ") -> o: 
  "summarize numbers"
  return o(it=Num, at=at, txt=txt, lo=1e32, mu=0, m2=0, sd=0, n=0, 
           hi=-1e32, more = 0 if txt[-1] == "-" else 1)

def Data(src:iter) -> o:
  "store rows, summarized in cols"
  src = iter(src)
  return adds(src, o(it=Data, n=0, rows=[], cols= Cols(next(src))))

def Cols(names: list[str]) -> o:
  "generate columns from column names"
  all, x, y, klass = [],[],[],None
  for c,s in enumerate(names):
    all += [(Num if s[0].isupper() else Sym)(c,s)]
    if s[-1] == "X": continue
    if s[-1] == "!": klass = all[-1]
    (y if s[-1] in "!-+" else x).append(all[-1])
  return o(it=Cols, names=names, all=all, x=x, y=y, klass=klass)

def clone(data:Data, rows=None) -> o:
  "copy data structure, maybe add in rows"
  return adds(rows or [], Data([data.cols.names]))

#--------------------------------------------------------------------
def adds(src:iter, it=None) ->o:
  "add many things to it, return it"
  it = it or Num()
  [add(it, x) for x in src]
  return it

def sub(x, v, zap=False): 
  "subtraction is just adding -1"
  return add(x,v,-1,zap)

def add(x: o, v:any, inc=1, zap=False):
  "incrementally update Syms,Nums or Datas"
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
def dist(src) -> float:
  "general distance function"
  d,n = 0,0
  for v in src: n,d = n+1, d + v**the.p
  return (d/n) ** (1/the.p)

def disty(data:Data, row:Row) -> float:
  "distance of row to best goal values"
  return dist(abs(norm(c, row[c.at]) - c.more) for c in data.cols.y)

def distysort(data,rows=None) -> list[list]:
  "sort rows by distance to best goal values"
  return sorted(rows or data.rows, key=lambda r: disty(data,r))

def norm(num:Num, v:float) -> float:  # 0..1
  "map 'v' to range 0..1"
  return  (v - num.lo) / (num.hi - num.lo + 1E-32)

#--------------------------------------------------------------------
def csv(file:str) -> list[Row]:
  "iterate over a file"
  with open(file,encoding="utf-8") as f:
    for line in f:
      if (line := line.split("%")[0]):
        yield [atom(s.strip()) for s in line.split(",")]

def shuffle(lst: list) -> list:
  "randomize order of list"
  random.shuffle(lst)
  return lst

def pout(x:any) -> None: 
  print(out(x))

def out(x:any) -> str:
  "pretty print anything"
  if callable(x): x= x.__name__
  if type(x) is float: x = int(x) if int(x)==float(x) else f"{x:.3f}"
  if hasattr(x,"__dict__"):
    x= "{" + ' '.join(f":{k} {out(v)}" 
             for k,v in x.__dict__.items() if str(k)[0] != "_") + "}"
  return str(x)

def main(funs: dict[str,callable]) -> None:
  "from command line, update config find functions to call"
  for n,arg in enumerate(sys.argv):
    if (fn := funs.get(f"eg{arg.replace('-', '_')}")):
      random.seed(the.seed); fn()
    else:
      for key in vars(the):
        if arg=="-"+key[0]: the.__dict__[key] = atom(sys.argv[n+1])

#--------------------------------------------------------------------
def like(i:o, v:any, prior=0) -> float :
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
  prior = (data.n + the.k) / (nall + the.k*nh)
  tmp = [like(col,v,prior) 
         for col in data.cols.x if (v:=row[col.at]) != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)    

def likely1(best:Data, rest:Data, x:Data) -> Row:
  "Remove from `x' any 1 thing more best-ish than rest-ish."
  shuffle(x.rows)
  j, nall = 0, best.n + rest.n
  for i,row in enumerate(x.rows[:the.Few*2]):
    if likes(best,row,nall,2) > likes(rest,row,nall,2):
      j = i; break
  return x.rows.pop(j)

def likelier(best:Data, rest:Data, x:Data) -> Row:
  "Sort 'x by the.acq, remove first from 'x'. Return first."
  e, nall = math.e, best.n + rest.n
  p = nall/the.Build
  q = {'xploit':0, 'xplor':1}.get(the.acq, 1-p)
  def _fn(row):
    b,r = e**likes(best,row,nall,2), e**likes(rest,row,nall,2)
    if the.acq=="bore": return b*b/(r+1e-32)
    return (b + r*q) / abs(b*q - r + 1e-32)

  first, *lst = sorted(x.rows[:the.Few*2], key=_fn, reverse=True)
  x.rows = lst[:the.Few] + x.rows[the.Few*2:] + lst[the.Few:] 
  return first

def likely(data:Data, rows=None) -> list[Row]:
  """x,xy = rows with 'x' and 'xy' knowledge.
  Find the thing in x most likely to be best. Add to xy. Repeat."""
  rows = rows or data.rows
  x   = clone(data, shuffle(rows[:]))
  xy, best, rest = clone(data), clone(data), clone(data)
  # label anything
  for _ in range(the.Any):
    add(xy, sub(x, x.rows.pop()))
  # divide lablled items into best and rest
  xy.rows = distysort(xy)
  n = round(the.Any**.5)
  adds(xy.rows[:n], best)
  adds(xy.rows[n:], rest)
  # loop
  fn = likely1 if the.acq=="klass" else likelier
  while x.n > 2 and xy.n < the.Build:
    add(xy, add(best, sub(x, fn(best, rest, x))))
    if best.n > (xy.n**.5):
      best.rows = distysort(xy,best.rows)
      while best.n > (xy.n**.5):
        add(rest, sub(best, best.rows.pop(-1)))
  return distysort(xy)
 
#--------------------------------------------------------------------
def eg_h(): 
  "show help"
  print(__doc__)

def eg__the(): 
  "show config"
  print(the)

def eg__sym(): 
  "Sym:  demo"
  assert 4==adds("aaaabbc",Sym()).has["a"]

def eg__Sym(): 
  "Sym:  demo symbolic likelihood calcs"
  s = adds("aaaabbc",Sym()); assert 0.44 == round(like(s,"a"),2)

def eg__num(): 
  "Num: check Num sample tacks gaussians"
  assert 10==round(adds(random.gauss(10,2) for _ in range(1000)).mu,1)

def eg__Num() : 
  "Num: demo nuneric likelihood calcs"
  n = adds(random.gauss(10,2) for _ in range(1000))
  assert 0.13 == round(like(n,10.5),2)

def eg__data():
  assert 3009.84 == round(sum(y.mu for y in 
                              Data(csv(the.file)).cols.y),2)

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
    if d2.n==100:  
      mu1,sd1 = x.mu,x.sd ; print("asIs",mu1,sd1)
  for row in d1.rows[::-1]:
    if d2.n==100: 
      mu2,sd2 = x.mu,x.sd ; print("toBe",mu2,sd2)
      assert abs(mu2 - mu1) < 1.01 and abs(sd2 - sd1) < 1.01
    sub(d2,row,zap=True)

def eg__likely():
  data = Data(csv(the.file))
  b4   = adds(disty(data,r) for r in data.rows)
  R    = lambda n: int(100*n)
  win  = lambda n: R((1 - (n - b4.lo) / (b4.mu - b4.lo)))
  rxs  = dict(klass=Num(),xploit=Num(),xplor=Num(),adapt=Num())
  for acq,log in rxs.items():
    the.acq = acq
    adds((disty(data, likely(data)[0]) for _ in range(3)), log)
  zero=rxs["klass"]
  print(*map(win,[zero.mu] + [log.mu
                         for acq,log in rxs.items() if acq != "klass"]),
        the.file)

def eg__all(): 
 for s,fn in globals().items():
   if s != "eg__all" and s.startswith("eg_"): 
     print(f"\n--| {s} |--------------"); random.seed(the.seed); fn()


#--------------------------------------------------------------------
if __name__ == "__main__": main(globals())
