# Why? What?

Recently, AI has gotten very complicated.  The models themselves
are opaque and  hard to understand or audit or repair.  The CPU
required to build and use them severely limits experimentation. It
also complicates industrial deployment and teaching.

But are all problems inherently complex and need, say, generative
AI and large language models?  _Tiny AI_ performs _predictive AI
from small models_ and handles tasks like optimization and regression.
As shown here, tiny AI can be remarkable effective, yet simple to
code, and need only a few dozen labeled examples for training.

Tiny AI methods are routinely ignored in research and industry.  In
a recent systematic review [^hou24] of 229 SE papers using large
language models (LLMs), only 13/229 (about 5%) of those papers
compared LLMs to other approaches. This is a methodological error
since other methods can produce results that are better and/or
faster (See Table 1)


> Table1: Predictive AI can sometimes produce better results, faster.


|When          | What|
|--------------|-----|
|2018 [^maju18] | Simple clustering plus predictive AI did better for text mining. | 
|2022 [^grin22] | Large language models may not be the best choice for tabular data. |
|2024 [^somv24] | Ditto |
|2022 [^taw23] | Predictive AI did better for management for agile software development. |
|2024 [^ling24] | Predictive AI did better for data synthesis. |
| 2024 [^john24] | Long list of errors seen in generative AI for software engineering. |

Perhaps the reason Tiny AI is ignored is that there is no simple
reference package, nor documentation of its effectiveness.  To
remedy that, we offer a free open source Tiny AI  python package, accessible via

     pip install ezr

EZR is an explanation system for incremental multi-objective
optimization.  This tool sorts and splits the examples seen so far
into two lists: a small "best" list and the remaining "rest".  New
examples are explored if they are more likely to be "best" than "rest".
The most preferred example then updates "best" and "rest" and the
cycle repeats.  At runtime, EZR avoids data that is noisy (i.e.
is clearly not "best" or "rest") and  superfluous  (i.e. that is not
relevant for "better" behavior). In this way EZR ignores most of the
data and builds its models using just a few dozens samples. Hence,
a regression tree learned from these examples offers a tiny and simple explanation
of how to achieve good results (and  also what to do to improve
those results).

EZR is very short (a few hundred lines of Python; no use of complex
packages like pandas or scikit-learn). 
EZR has been tested on over  100 example problems from the
recent search-based SE literature [^moot].  Those problems are as varied
as minimizing cost, defects, and development time while maximizing
functionality; tuning data‑miner settings like the number of trees
in a random forest; predicting open‑source project health; and
optimizing software projects or cloud configurations. Beyond software
engineering, these problems also including selecting football teams, retraining
employees, reducing school dropouts, approving loans, predicting life
expectancy or disease spread, designing cars, choosing wines, and even
planning winning ad campaigns. 

Across all the problems, EZR usually performs
very well at finding good solutions
after sampling just a few dozen examples.
 We offer it here as

- A useful tool for teaching AI and SE and scripting;
- A productive tool for conducting state-of-the-art resaerch. 
- A criticism of other work that has never checked complicated a simple idea; 


For an example where this tool can dramatically simplify prior results, see the end of this document.

[^moot]: http://github.com/timm/moot

[^hou24]: Hou, X., Zhao, Y., Liu, Y., Yang, Z., Wang, K., Li, L., ... & Wang, H. (2024). Large language models for software engineering: A systematic literature review. ACM Transactions on Software Engineering and Methodology, 33(8), 1-79.
[^john24]: Johnson, B., & Menzies, T. (2024). Ai over-hype: A dangerous threat (and how to fix it). IEEE Software, 41(6), 131-138.

[^ling24]: Ling, X., Menzies, T., Hazard, C., Shu, J., & Beel, J. (2024). Trading off scalability, privacy, and performance in data synthesis. IEEE Access, 12, 26642-26654.

[^grin22]: Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. "Why do tree-based models still outperform deep learning on typical tabular data?." Advances in neural information processing systems 35 (2022): 507-520

[^somv24]: Somvanshi, S., Das, S., Javed, S. A., Antariksa, G., & Hossain, A. (2024). A survey on deep tabular learning. arXiv preprint arXiv:2410.12034.

[^maju18]: Majumder, S., Balaji, N., Brey, K., Fu, W., & Menzies, T. (2018, May). 500+ times faster than deep learning: A case study exploring faster methods for text mining stackoverflow. In Proceedings of the 15th International Conference on Mining Software Repositories (pp. 554-563).

[^taw23]: V. Tawosi, R. Moussa, and F. Sarro, “Agile effort
estimation: Have we solved the problem yet? insights from a replication
study,” IEEE Transactions on Software Engineering, vol. 49, no. 4,
pp. 2677– 2697, 2023.

## A Quick Example

Just to give this work some context, here’s a concrete case.

Say we want to configure a database to reduce energy use, runtime,
and CPU load. The system exposes dozens of tuning knobs—storage,
logging, locking, encryption, and more. Understanding how each
setting impacts performance is daunting.

When manual reasoning fails, we can ask AI to help.  Imagine we
have a log of 800+ configurations, each showing the measured effects
of settings to dozens of control settings (shown here as x1,x2,x3...).
Some settings lead to excellent results:

```
choices                     Effects
----------------            -----------------------
x1,x2,x3...   →            Energy-, time-,  cpu-
0,0,0,0,1,0,...               6.6,    248.4,   2.1   ← best
1,1,0,1,1,1,...              16.8,    518.6,  14.1   ← rest
...
```

We say the better examples are those that are "closer to heaven";
i.e.  if each example achieves goals $g1,g2,...$; and the best
values ever seen for  each goal is $n1,n2,..$; then distance to
heaven is the Euclidean distance to the best values:

$$y= \sqrt{\left(\sum_i N(abs(g_i-n_i))^2\right) / len(goals)}$$

where `N` normalizes our goals values min..max as 0..1  The closer to heaven,
the better the example so we say _smaller_ $y$ values are _better_.
 To simplify the reporting, we define _optimal_ to
be the labeled example that is closest to heaven (i.e. has the smallest $y$ values).
If $\hat{y}$ is the mean $y$ of all the rows, and $y_0$ comes from the optimal
row, and our optimizer returns a row with a  score $y_1$ then the  _win_
of that estimation is the normalized distance from mean to best:

$$win = 100\left(1- \frac{y_1 - y_0}{\hat{y}-y_0}\right)$$

Note that a win of 100 means "we have reached the optimal" and a win less than 0 means
an optimization failure (since we are finding solutions worse than before.

Using $y$, a list of examples seen-so-far can be sorted into
a small "best" set and a larger "rest" set.
Any number of AI tools could learn what separates “best” from “rest.”
But here's the challenge: **labeling** each configuration (e.g.,
by running all the benchmarks for all possible configurations) is
expensive. So the EZR challenge is how to learn an effective model
with minimal effort?

To handle that challenge,  EZR uses a minimalist A–B–C strategy:

- **A=Any**; i.e. "ask anything".
  Randomly label a few examples (say, _A = 4_) to seed the process.

- **B=Build**; i.e.  build a model**
  In this phase, we build separate models for “best” and “rest,” then label up to,
  say _B = 24_ more rows by picking the unlabeled row that maximizes
  the score _b/r_ (where `b` and `r` are likelihoods that a row
  belongs to the "best" and "rest" models).

- **C=Check**; i.e. check the model.
  Apply the learned model to unlabeled data and to select a small
  set (e.g., _C=5_) of the most promising rows. After labeling
  those rows, return the best one.

In this task, after labeling just 24 out of 800 rows (∼3%), EZR
constructs a binary regression tree from those 24 examples. In that
tree, left and right branches go best and worse examples. The
left-most branch of that tree is shown here (and to get to any line
in this tree, all the things above it have to be true).

    if crypt_blowfish == 0
    |  if memory_tables == 1
    |  |  if detailed_logging == 1
    |  |  |  if no_write_delay == 0; <== win=98%
    
These four conditions select rows that very clone
(98%) to  optimal.

Note that this branch only mentions four
options, and two of those are all about what to turn off. That
is to say, even though this databased has dozens of configuration
options, there are two bad things to avoid and only two most
important thing to enable  (_memory\_tables_ and _detailed\_logging_).

Of course, if you ever show a result like this to a subject
matter expert, they will  push back. For example, they might demand
to know what happens when `crypt_blowfish` is enabled. Blowfish in
password hashing scheme.  It makes password protection slower but
it also  increases the computational effort required for attackers
to  brute-force attack the database's security.  The full tree
generted by EZR shows what happens this feature is enabled.
(see the last two lines).
Note
all the negative "wins" which is to say, if your goals are fast
runtimes, do not `crypt_blowfish`.

     #rows  win
        17   68    if crypt_blowfish == 0
         7   94    |  if memory_tables == 1
         5   97    |  |  if detailed_logging == 1
         4   98    |  |  |  if no_write_delay == 0;
        10   49    |  if memory_tables == 0
         7   51    |  |  if encryption == 0
         5   50    |  |  |  if no_write_delay == 1
         4   50    |  |  |  |  if txc_mvlocks == 0;
         7 -165    if crypt_blowfish == 1
         4  -51    |  if memory_tables == 1;

(Aside: of course if security is an important goal then (a) add a
"security+" column to the training data shown above; (b)  re-un
EZR; (c) check what are the mpracts of that additional goal.)

EZR shows that with the right strategy, a handful of examples 
(in this case, 24) can
uncover nearly all the signal.  All of this took just a few dozen
queries—and a few hundred lines of code. It’s a striking illustration
of the Pareto principle:  **most of the value often comes from just
a small fraction of the effort**.



## Simp
is only true for generative AI. For predictive AI, as shown here,

This code implements an AI agent explaining what they found as they
incrementally explore the world, updating their 
knowledge of what is best or rest.
As it walks the world, our agent finds rows of data
which have  one or more `x` independent  values
and one or more `y` goals (which must be mininized or maximized).

Our agent knows that 
getting `x` values is usually much faster and cheaper than getting `y` values.
For example,
in a  single glance, we can find all the colors of all the cars in  a used car lot.
But it takes hours of driving per car to determine their mileage.




This code is reversed engineers from decades of experiments by dozens
of graduate students exploring data science. 

To say that another way, this is an explanation algorithm
for active learning for
for multi-objective optimization.

Here's the core idea: our agent randmoly
walks around the world keeping a list
of best and rest things seen so far. If they find a new best thing,
then they revise  what it means to be "best":

```
best, rest = [], all
shuffle(rest)
 while rest not empty:
   if model(new_item := pop(rest)) says "best":
     push(best, new_item)
     rebuild_model(best, rest)  
```

This short description misses many improtant details. 

Firstly, what are we exploring? This code processes rows of data each of Firstly, what do we mean by "best"? We say the better examples
are those that are "closer to heaven"; i.e.
if each example achieves goals $g1,g2,...$; and 
the best valuse ever seen for  each goal is $n1,n2,..$; then

$$y= \sum_i(N(abs(g_i-n_i))^2) / len(goals)$$

where `N` normalizes our goals values 0..1.
The columns of these rows have names:
upper case names are for numerics; and anything marked with `+` or `-` is a goal to be maximzed
or minimized. 


to get started, we need to build an initial model.

```
Any=4    # inota;;y. ;abel Any items
Build=24 # labelling budget used when building models
Check= 5 # 

shuffle(all)
n= len(all)//2
train,test = all[:n],all[n:]

dark = train[Any:] 
lite = [label(x) for x in train[:Any]
lite.sort(y)
best, rest = lite[:Any/2], lite[Any//2:]

tooBig(best) -> len(best) > sqrt(len(best) + len(rest))
label(row)   -> add_Y_labels(row); return row
error(model,row) -> float  

while (Build--) > 0:
  model = train(best,rest)
  push(best, (new := label(pop(dark))))
  if tooBig(best):
    best.sort(row -> error(model, row))
    while tooBig(best):
      # move the worst best into rest
      push(rest, best.pop(-1)) 
best.sort(row -> error(model,row)) # ensure best is sorted
model = train(best,rest) 
# sort test on model, check the best guesses
return test.sort(row -> error(model,row))[:Check]
```

