#!/usr/bin/env python3 -B
import sys,re
sys.path += ["../src"]
from abc import *

def all_egs(run=False):
  "Run all eg__* functions."
  for k,fn in globals().items():
    if k.startswith('eg__') and k != 'eg__all':
      if run:
        print("\n----["+k+"]"+'-'*40)
        random.seed(the.seed)
        fn()
      else:  
        print(" "+re.sub('eg__','--',k).ljust(10),"\t",fn.__doc__ or "")

#  _        _.  ._ _   ._   |   _    _
# (/_  ><  (_|  | | |  |_)  |  (/_  _> 
#                      |               

def eg_h()    : print(__doc__)
def eg__all() : all_egs(run=True)
def eg__list(): all_egs()

def eg__the(): print(the)
def eg__sym(): print(adds("aaaabbc"))
def eg__Sym(): s = adds("aaaabbc"); assert 0.44 == round(like(s,"a"),2)
def eg__num(): print(adds(random.gauss(10,2) for _ in range(1000)))
def eg__Num() : 
  n = adds(random.gauss(10,2) for _ in range(1000))
  assert 0.13 == round(like(n,10.5),2)

def eg__data(): [print(col) for col in dataRead(the.file).cols.all]

def eg__inc():
  "Check i can add/delete rows incrementally."
  d1 = dataRead(the.file)
  d2 = dataClone(d1)
  x  = d2.cols.x[1]
  for row in d1.rows:
    dataAdd(d2,row)
    if len(d2.rows)==100:  
      mu1,sd1 = x.mu,x.sd 
      print(mu1,sd1)
  for row in d1.rows[::-1]:
    if len(d2.rows)==100: 
      mu2,sd2 = x.mu,x.sd
      print(mu2,sd2)
      assert abs(mu2 - mu1) < 1.01 and abs(sd2 - sd1) < 1.01
      break
    dataAdd(d2,row,inc=-1,zap=True)

def eg__bayes():
  data = dataRead(the.file)
  assert all(-30 <= likes(data,t) <= 0 for t in data.rows)
  print(sorted([round(likes(data,t),2) for t in data.rows])[::20])

def eg__confuse():
  "check confuse calcs."
  # a b c <- got
  # ------. want
  # 5 1   | a
  #   2 1 | b
  #     3 | c
  cf = Confuse()   
  for want,got,n in [
      ("a","a",5),("a","b",1),("b","b",2),("b","c",1),("c","c",3)]:
    for _ in range(n): confuse(cf, want, got)
  xpect = {"a": {'pd':83,  'acc':92, 'pf':0,  'prec':100},
           "b": {'pd':67,  'acc':83, 'pf':11, 'prec':67},
           "c": {'pd':100, 'acc':92, 'pf':11, 'prec':75} }
  for y in confused(cf):
    if y.label != "_OVERALL":
       got = {'pd':y.pd, 'acc':y.acc, 'pf':y.pf, 'prec':y.prec}
       assert xpect[y.label] == got
  show(confused(cf))

def eg__stats():
   b4 = [random.gauss(1,1)+ random.gauss(10,1)**0.5
         for _ in range(20)]
   d, out = 0,[]
   while d < 1:
     now = [x+d*random.random() for x in b4]
     out += [f"{d:.2f}" + ("y" if statsSame(b4,now) else "n")]
     d += 0.05
   print(', '.join(out))

def daRx(t):
    if not isinstance(t,(tuple,list)): return str(t)
    return ':'.join(str(x) for x in t)

def eg__sk():
  n=20
  for sd in [0.1,1,10]:
    for eps in [1E-32,0.05,0.1,0.15,0.2]:
      print("\neps=",eps, "sd=",sd)
      rxs={}
      G=lambda m:[random.gauss(m,sd) for _ in range(n)]
      for i in range(20): 
        if   i<=  4 : rxs[chr(97+i)] = G(10)
        elif i <= 8 : rxs[chr(97+i)] = G(11)
        elif i <=12 : rxs[chr(97+i)] = G(12)
        elif i <=16 : rxs[chr(97+i)] = G(12)
        else        : rxs[chr(97+i)] = G(14)
      out=statsRank(rxs,eps=eps)
      print("\t",''.join(map(daRx,out.keys())))
      print("\t",''.join([str(x) for x in out.values()]))

def eg__diabetes(): 
  show(likeClassifier("../moot/classify/diabetes.csv"))

def eg__soybean():  
  show(likeClassifier("../moot/classify/soybean.csv"))

def eg__distx():
  data = dataRead(the.file)
  r1= data.rows[0]
  ds= sorted([distx(data,r1,r2) for r2 in data.rows])
  print(', '.join(f"{x:.2f}" for x in ds[::20]))
  assert all(0 <= x <= 1 for x in ds)

def eg__disty():
  data = dataRead(the.file)
  data.rows.sort(key=lambda row: disty(data,row))
  assert all(0 <= disty(data,r) <= 1 for r in data.rows)
  print(', '.join(data.cols.names))
  print("top4:");   [print("\t",row) for row in data.rows[:4]]
  print("worst4:"); [print("\t",row) for row in data.rows[-4:]]

def eg__irisKpp(): 
  [print(r) for r in distKpp(dataRead("../moot/classify/iris.csv"),k=10)]

def eg__irisK(): 
  for data in distKmeans(dataRead("../moot/classify/iris.csv"),k=10):
    print(mids(data)) 

def daBest(data,rows=None):
  rows = rows or data.rows
  Y=lambda r: disty(data,r)
  return Y(sorted(rows, key=Y)[0])

def eg__c():
  data = dataRead(the.file)
  n = len(data.rows)//2
  repeats= 20
  for i in range(repeats):
    random.shuffle(data.rows)
    rows1, rows2 = data.rows[:n], data.rows[n:]
    data1, data2 = dataClone(data, rows1), dataClone(data, rows2)
    like1 = likely(data1, rows1, "klass")
    tree1 = Tree(dataClone(data,like1.labels))
    #if i == 0: 
    #  treeShow( Tree(dataClone(data,like1.labels)))
    def gt(row):
      nall = len(like1.best.rows)+len(like1.rest.rows)
      b    = likes(like1.best, row, nall, 2) 
      r    = likes(like1.rest, row, nall, 2)
      return b - r
    rows2leaf  = sorted(rows2, key=lambda r: treeLeaf(tree1,r).ys.mu)[:the.Check]
    rows2bayes = sorted(rows2, key=gt, reverse=True)[:the.Check]
    print(f"{daBest(data2, rows2leaf):.3f} {daBest(data2, rows2bayes):.3f}")
  
def eg__tree():
  "XXX: extend using best rest to select"
  data = dataRead(the.file)
  n = len(data.rows)//2
  repeats= 10
  for i in range(repeats):
    if i==0: treeShow(Tree(dataClone(data, likely(data,data.rows,"klass").labels)))
    random.shuffle(data.rows)
    train, test = dataClone(data, data.rows[:n]), dataClone(data, data.rows[n:])

    rx0   = random.choices(test.rows,k=the.Check)
    tree  = Tree(dataClone(data, likely(train,train.rows,"klass").labels))
    tree1 = Tree(dataClone(data, random.choices(train.rows,k=the.Build*10)))
    rx    = sorted(test.rows, key=lambda r: treeLeaf(tree,r).ys.mu)
    rx1   = sorted(test.rows, key=lambda r: treeLeaf(tree1,r).ys.mu)

    base = adds(disty(test,r) for r in test.rows)
    win  = (1 - (daBest(test,rx[:the.Check])  - base.lo)/(base.mu - base.lo))
    win0 = (1 - (daBest(test,rx0[:the.Check]) - base.lo)/(base.mu - base.lo))
    win1 = (1 - (daBest(test,rx1[:the.Check]) - base.lo)/(base.mu - base.lo))
    diff = 0 if abs(win - win1) < .35*base.sd else (win-win1)
    R    = lambda z:int(100*z)
    print(R(.35*base.sd), "tree", R(win), "tree10", R(win1), "rand", R(win0) , "diff", R(diff))

def eg__fmap():
  data = dataRead(the.file)
  for few in [32,64,128,256,512]:
    the.Few = few
    print(few)
    n=adds(daBest(data, distFastermap(data,data.rows).labels.rows) for _ in range(20))
    print("\t",n.mu,n.sd)

def eg__acq():
  print(1)
  data = dataRead(the.file)
  base = adds(disty(data,r) for r in data.rows)
  for few in [15,30,60]:
    the.Few = few
    print(few)
    for acq in ["klass"]: #["xplore", "xploit", "adapt","klass"]:
      the.acq = acq
      n=adds(daBest(data, likely(data, data.rows).labels) for _ in range(20))
      print(f"\t {the.acq} {base.mu:.3f}, {base.lo:.3f} {n.mu:.3f} {n.sd:.3f}")

def eg__rand():
  data = dataRead(the.file)
  n = adds(daBest(data, random.choices(data.rows, k=the.Build)) for _ in range(20))
  print("\t",n.mu,n.sd)

def eg__old():
  data = dataRead(the.file)
  rxs = dict(
             distKpp   = lambda d: distKpp(d, d.rows),
             sway   = lambda d: distFastermap(d, d.rows, sway1=True).labels.rows,
         sway2  = lambda d: distFastermap(d, d.rows).labels.rows # <== winner #<== best
             )
  xper1(data,rxs)
# 77,  88, 96
#  195  #file                                       rows    |y|  |x|   asIs  min  distKpp:15  sway:15  sway2:15  distKpp:30  sway:30  sway2:30  distKpp:60  sway:60  sway2:60  distKpp:120  sway:120  sway2:120  win A
#  196  13                                                                         35      36       35        35      38       43        35      38       70        34       39        100           A
             
def eg__liking():
  data = dataRead(the.file)
  rxs = dict(#rand   = lambda d: random.choices(d.rows,k=the.Build),
             klass = lambda d: likely(d,d.rows,"klass").labels, #<== klass
             xploit = lambda d: likely(d,d.rows,"xploit").labels,
             xplor = lambda d: likely(d,d.rows,"xplor").labels, 
           adapt  = lambda d: likely(d,d.rows,"adapt").labels
             )
  xper1(data,rxs)

# win percentils 25,50,75 =  76, 84,95 
# #file,rows,|y|,|x|,asIs,min,klass:15,xploit:15,xplor:15,adapt:15,klass:30,xploit:30,xplor:30,adapt:30,klass:60,xploit:60,xplor:60,adapt:60,klass:120,xploit:120,xplor:120,adapt:120,win,A
# 13,,,,32,30,31,31,36,33,34,34,64,45,48,50,96,66,72,73,A


def eg__final():
  data = dataRead(the.file)
  rxs = dict(rand   = lambda d: random.choices(d.rows,k=the.Build),
             klass = lambda d: likely(d,d.rows,"klass").labels, #<== klass
             sway2  = lambda d: distFastermap(d, d.rows).labels.rows # <== sway2
             )
  xper1(data,rxs)

# win percentils 25,50,75 =  78, 87,96 
   # 199  #file                                       rows    |y|  |x|   asIs  min  rand:15  klass:15  sway2:15  rand:30  klass:30  sway2:30  rand:60  klass:60  sway2:60  rand:120  klass:120  sway2:120  win A
   # 200  13                                                                         30       30        33        39       35        42        59       58        65        82        83         93            A
fyi=lambda s: print(s,file=sys.stderr, flush=True,end="")

def xper1(data,rxs):
  repeats=30
  builds=[15,30,60,120]
  base = adds(disty(data,r) for r in data.rows)
  win  = lambda x: 1 - (x - base.lo) / (base.mu - base.lo + 1e-32)
  out={}
  for build in builds: 
    the.Build = build
    fyi("+")
    for rx,fn in rxs.items():
      fyi("-")
      out[(rx,build)] = [daBest(data,fn(data)) for _ in range(repeats)]
  fyi("!")
  ranks = statsRank(out, eps=base.sd*0.2)
  rank1 = adds(x for k in ranks if ranks[k] == 1 for x in out[k])
  p = lambda z: round(100*z) #"1.00" if z == 1 else (f"{pretty(z,2)[1:]}" if isinstance(z,float) and z< 1 else str(z))
  q = lambda k: f" {'A' if ranks[k]==1 else ' '} {p(adds(out[k]).mu)}"
  print("#file","rows","|y|","|x|","asIs","min",*[daRx((rx,b)) for b in builds for rx in rxs],"win",sep=",")
  print(re.sub("^.*/","",the.file),
        len(data.rows), len(data.cols.y), len(data.cols.x), p(base.mu), p(base.lo),
        *[q((rx,b)) for b in builds for rx in rxs],p(win(rank1.mu)), sep=",")

#   __                                  
#  (_   _|_   _.  ._  _|_  __       ._  
#  __)   |_  (_|  |    |_      |_|  |_) 
#                                  |   
#-----------------------------------------------------------
print(the)
if __name__ == "__main__":
  for n,arg in enumerate(sys.argv):
    if (fn := globals().get(f"eg{arg.replace('-', '_')}")):
      random.seed(the.seed)
      fn() 
    else:
      for key in vars(the):
        if arg == "-"+key[0]: 
          the.__dict__[key] = coerce(sys.argv[n+1])
