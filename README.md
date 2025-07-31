Here's the core idea: our agent randmoly
walks around the world keeping a list
of best and rest things seen so far. If they find a new best thing,
then they revise  what it means to be "best":


   best, rest = [], all
   shuffle(rest)
    while rest not empty:
      if model(new_item := pop(rest)) says "best":
        push(best, new_item)
        rebuild_model(best, rest)  

This short description misses many improtant details. 

Firstly, we wills ay best means "distance to heaven"; i.e.
if each example achieves goals $g1,g2,...$ and 
the best valuse for each goal are $n1,n2,..$ then

$$y= \sigma_i(N(abs(g_i-n_i))^2) / len(goals)$$

where `N` normalizes our goals values 0..1.

to get started, we need to build an initial model.

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
    
