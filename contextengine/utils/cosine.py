import math

def cosine_similarity(a:list,b:list)->float:
    
    dot=sum(x*y for x,y in zip(a,b))
    norm_a=math.sqrt(sum(x*x for x in a))
    norm_b=math.sqrt(sum(y*y for y in b))
    return dot/(norm_a * norm_b + 1e-9)

