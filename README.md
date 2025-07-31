Teaching and researeching AI has gotten complcaited. 
The models themselves are opaque and  hard to understand or audit or repair.
The CPU required to build and use them severely
limits experimentation.

Certainly, some probems are intherently complex and need, say, generative AI and large language models.
But not all AI is generative AI.
_Tiny AI_ performs _predictive AI from small models_.
As shown here,
predictive AI can be remarkable effective, yet simple to code, and need only a few dozen
labelled examples. to train. 

Tiny AI methods are routinely ignored in resaerch and industry.
In a 
recent systematic review [^hou24] of 229 SE papers using large
language models (LLMs), only 13/229 (about 5%) of those papers
compared LLMs to other approaches. This is a methodological error since other methods can produce results that are better
and/or faster (See Table 1)

Ideally researechers and practioners first try Tiny AI methods, then move to Big AI only when necessary.
In an attempt to encourage that methodology, we offer here an example of how Tiny AI can be simple, fast, and effective. No,
it this code will not solve all problems. But since it is so non-resource intensive, we comreccond it as a first pass.


[^hou24]: Hou, X., Zhao, Y., Liu, Y., Yang, Z., Wang, K., Li, L., ... & Wang, H. (2024). Large language models for software engineering: A systematic literature review. ACM Transactions on Software Engineering and Methodology, 33(8), 1-79.

> Table1: Predictive AI can sometimes produce better results, faster.

|When          | What|
|--------------+-----|
|2022 [^grin22] | Large language models may not be the best choice for tabular data. |
|2024 [^somv24] | Ditto |
|2022 [^tawosi] | Predictive AI did better for management for agile software development. |
|2018 [^maju18] | Simple clustering plus predictive AI did better for text mining. | 
|2024 [^ling24] | Predicve AI did better for data synthesis. |
| 2024 [^john24] | Long list of errors seen in generataive AI for softwaer engineering. |

[^john24]: Johnson, B., & Menzies, T. (2024). Ai over-hype: A dangerous threat (and how to fix it). IEEE Software, 41(6), 131-138.

[^ling24]: Ling, X., Menzies, T., Hazard, C., Shu, J., & Beel, J. (2024). Trading off scalability, privacy, and performance in data synthesis. IEEE Access, 12, 26642-26654.

[^grin22]: Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. "Why do tree-based models still outperform deep learning on typical tabular data?." Advances in neural information processing systems 35 (2022): 507-520

[^sonv24]: Somvanshi, S., Das, S., Javed, S. A., Antariksa, G., & Hossain, A. (2024). A survey on deep tabular learning. arXiv preprint arXiv:2410.12034.

[^maju18]: Majumder, S., Balaji, N., Brey, K., Fu, W., & Menzies, T. (2018, May). 500+ times faster than deep learning: A case study exploring faster methods for text mining stackoverflow. In Proceedings of the 15th International Conference on Mining Software Repositories (pp. 554-563).


Lest we forget the valye of predictive AI, this paper offers an example of tiny AI called EZR.
EZR is very short (a few hundred lines).
We offer it here as
(a) a criticism of much other work
that has never chec complicated a simple idea; (b) a useful tool for teaching AI and SE and scripting;
(c) a productive tool for conducting state of the art resaerch. 
For an example where this tool can dramatically simplfy prior resutls, see the end of this document.


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

