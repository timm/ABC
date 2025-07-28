#!/usr/bin/env python3 -B
"""
abc101.py, simple version of abc. acq=klass only
(c) 2025, Tim Menzies <timm@ieee.org>, MIT license

 -A  Any=4           on init, how many initial guesses?
 -B  Build=24        when growing theory, how many labels?
 -C  Check=5         when testing, how many checks?
 -F  Few=128         when sub-sampling rows, how many to use?
 -k  k=1             Bayes hack for rare classes
 -l  leaf=2          min leaf size
 -m  m=2             Bayes hack for rare attributes
 -p  p=2             distance calculation coefficient
 -s  seed=1234567890 random number seed
 -f  file=../moot/optimize/misc/auto93.csv
                     path to CSV file
"""
import random, math, re

class o:
  def __init__(i, **d): i.__dict__.update(d)
  def __repr__(i): 
    fn = lambda v: f"{v:.3f}" if type(v) is float else v
    return '{'+' '.join(f":{k} {fn(v)}" for k,v in i.__dict__.items())+'}'

def coerce(s):
  for fn in [int,float]:
    try: return fn(s)
    except Exception as _: pass
  s = s.strip()
  return {'True':True,'False':False}.get(s,s)

the = o(**{k:coerce(v) for k,v in re.findall(r"(\w+)=(\S+)",__doc__)})

#--------------------------------------------------------------
def Sym(at=0,txt=""): 
  return o(it="Sym",at=at,txt=txt,has={})

def Num(at=0,txt=" "): 
  return o(it="Num", at=at, txt=txt, 
           lo=1e32, mu=0, m2=0, sd=0, n=0, hi=-1e32, 
           most = str(txt)[-1] !="-")

def Cols(names):
  all,x,y,klass = [],[],[],None
  for c,s in enumerate(names):
    all += [(Num if s[0].isupper() else Sym)(c,s)]
    if s[-1] != "X":
      if s[-1] == "!": klass= all[-1]
      (y if s[-1] in "!-+" else x).append(all[-1])
  return o(it="Cols",names=names, all=all, x=x, y=y, klass=None)

def Data(src):
  src = iter(src)
  return adds(src, o(it="Data", rows=[], cols=Cols(next(src))))

def dataClone(data,rows=[]): 
  return Data([data.cols.names] + rows)

#--------------------------------------------------------------
def adds(src, it=None):
  it = it or Num()
  [add(it,v) for v in src]
  return it

def sub(x, v, zap=False): return add(x,v,-1,zap)

def add(x, v, inc=1, zap=False):
  "Update a x with a value (and to subtract, use inc= -1)."
  if v == "?": return v
  if x.it == "Sym": x.has[v] = inc + x.has.get(v,0)
  elif x.it == "Num":
    x.n += inc
    x.lo, x.hi = min(v, x.lo), max(v, x.hi)
    if inc < 0 and x.n < 2:
      x.sd = x.m2 = x.mu = x.n = 0
    else:
      d     = v - x.mu
      x.mu += inc * (d / x.n)
      x.m2 += inc * (d * (v - x.mu))
      x.sd  = 0 if x.n < 2 else (max(0,x.m2)/(x.n-1))**.5
  elif x.it == "Data":
    if inc>0: x.rows += [v]
    elif zap: x.rows.remove(v) # slow for long rows
    [add(col, v[col.at],inc) for col in x.cols.all]
  return v

def norm(n, v): 
  return v if v=="?" or n.it=="Sym" else (v-n.lo)/(n.hi-n.lo + 1E-32)

def div(col):
  if col.it == "Num": return col.sd
  N = sum(col.has.values())
  return -sum(p*math.log(p,2) for n in col.has.values() if (p:=n/N) if n>0)

#--------------------------------------------------------------
def dist(src):
  d,n = 0,0
  for v in src: n,d = n+1, d + v**the.p
  return (d/n) ** (1/the.p)

def disty(data, row):
  return dist(abs(norm(c, row[c.at]) - c.most) for c in data.cols.y)

def distysort(data, rows=None):
  return sorted(rows or data.rows, key=lambda r: disty(data,r))


#--------------------------------------------------------------
def like(col, v, prior=0):
  if col.it =="Sym":
    out=((col.has.get(v,0) + the.m*prior)
         /(sum(col.has.values()) + the.m+1e-32))
  else:
    var= 2 * col.sd * col.sd + 1E-32
    z  = (v - col.mu) ** 2 / var
    out=  math.exp(-z) / (2 * math.pi * var) ** 0.5
  return min(1, max(0, out))

def likes(data, row, nall=100, nh=2):
  "How much does this DATA like row?"
  prior = (len(data.rows) + the.k) / (nall + the.k*nh)
  tmp = [like(col,v,prior) 
         for col in data.cols.x if (v:=row[col.at]) != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)    

def likely(data, rows):
  labels   = dataClone(data)
  nolabels = shuffle(rows[:])
  while len(nolabels) > 2 and len(labels.rows) < the.Build:
    if len(labels.rows) <= the.Any:
      add(labels, nolabels.pop())

    if len(labels.rows) == the.Any:
      labels.rows = distysort(labels)
      n = round(the.Any**0.5)
      best = dataClone(data, labels.rows[:n])
      rest = dataClone(data, labels.rows[n:])

    if len(labels.rows) >= the.Any:
      good, nolabels = likely1(best, rest,shuffle(nolabels))
      add(labels, add(best, good))

      if len(best.rows) >= len(labels.rows)**0.5:
        best.rows = distysort(best) 

      while len(best.rows) >= len(labels.rows)**0.5:
        add(rest, sub(best, best.rows.pop(-1)))

  return o(labels=distysort(labels), nolabels=nolabels,
           best=best, rest=rest)

def likely1(best, rest,  nolabels):
  good = nolabels[0]
  nall = len(best.rows) + len(rest.rows)
  for i, row in enumerate(nolabels[:the.Few*2]):
    if ( likes(best,row,2,nall) > likes(rest,row,2,nall) ):
      good = nolabels.pop(i); break
  return good, nolabels

#--------------------------------------------------------------
treeOps = {'<=' : lambda x,y: x <= y, 
           '=' : lambda x,y:x == y, 
           '>'  : lambda x,y:x > y}

def treeSelects(row,op,at,y): 
  "Have we selected this row?"
  return (x := row[at]) == "?" or treeOps[op](x, y)

def Tree(data, Y=None, how=None):
  "Create regression tree."
  Y = Y or (lambda row: disty(data, row))
  data.kids, data.how = [], how
  data.ys = adds(Y(row) for row in data.rows)
  if len(data.rows) >= the.leaf:
    hows = [how for col in data.cols.x 
            if (how := treeCuts(col,data.rows,Y))]
    if hows:
      for how1 in min(hows, key=lambda c: c.div).hows:
        rows1 = [r for r in data.rows if treeSelects(r, *how1)]
        if the.leaf <= len(rows1) < len(data.rows):
          data.kids += [Tree(dataClone(data,rows1), Y, how1)]
  return data

def treeCuts(col, rows,  Y):
  "Divide a col into ranges."
  def _sym(sym):
    d, n = {}, 0
    for row in rows:
      if (x := row[col.at]) != "?":
        n += 1
        d[x] = d.get(x) or Num()
        add(d[x], Y(row))
    return o(div = sum(c.n/n * div(c) for c in d.values()),
             hows = [("=",col.at,x) for x in d])
  
  def _num(num):
    out, b4, lhs, rhs = None, None, Num(), Num()
    xys = [(row[col.at], add(rhs, Y(row))) # add returns the "y" value
           for row in rows if row[col.at] != "?"]
    for x, y in sorted(xys, key=lambda z: z[0]):
      if x != b4 and the.leaf <= lhs.n <= len(xys) - the.leaf:
        now = (lhs.n * lhs.sd + rhs.n * rhs.sd) / len(xys)
        if not out or now < out.div:
          out = o(div=now, hows=[("<=",col.at,b4), (">",col.at,b4)])
      add(lhs, sub(rhs, y))
      b4 = x
    return out

  return (_sym if col.it == "Sym" else _num)(col)

def treeNodes(data, lvl=0, key=None):
  "iterate over all treeNodes"
  yield lvl, data
  for j in sorted(data.kids, key=key) if key else data.kids:
    yield from treeNodes(j,lvl + 1, key)

def treeLeaf(data, row):
  "Select a matching leaf"
  for j in data.kids or []:
    if treeSelects(row, *j.how): return treeLeaf(j,row)
  return data

def treeShow(data, key=lambda d: d.ys.mu):
  "Display tree (structure only, no numeric summaries)"
  ats = {}
  print(f"({data.ys.n})")
  for lvl, d in treeNodes(data, key=key):
    if lvl == 0: continue
    op, at, y = d.how
    name = data.cols.names[at]
    expl = f"if {name} {op} {y} :"
    indent = '|  ' * lvl
    if not d.kids:
      score = int(100 * (1 - (d.ys.mu - data.ys.lo) /
               (data.ys.mu - data.ys.lo + 1e-32)))
      leaf = f" {score} ({d.ys.n})"
    else:
      leaf = ''
    print(f"{indent}{expl}{leaf}")
    ats[at] = 1
  used = [data.cols.names[at] for at in sorted(ats)]
  print(len(data.cols.x), len(used), ', '.join(used))

#--------------------------------------------------------------
def csv(file):
  with open(file,encoding="utf-8") as f:
    for line in f:
      if (line := line.split("%")[0]):
        yield [coerce(s.strip()) for s in line.split(",")]

def shuffle(lst): random.shuffle(lst); return lst
