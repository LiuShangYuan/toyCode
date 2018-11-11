import Levenshtein
import numpy as np



### 构造KB树
def add_one(w, tree):
    '''
    tree = {'game':{1:{fame:{}}}}
    '''
    croot = list(tree.keys())[0] #当前树的根节点, 只有一个key
    d = Levenshtein.distance(w, croot)
    sub_tree = tree[croot].get(d, None)
    if sub_tree == None:
        tree[croot][d] = {w:{}}
    else:
        add_one(w, sub_tree)
        

### 使用KB树进行搜索
def fuzzy_search(n, q, bk_tree, rlist):
    '''
    q输入串
    n与输入串最大编辑距离
    '''
    if bk_tree is None:
        return
    root = list(bk_tree.keys())[0] ### 树根
    d = Levenshtein.distance(q, root)
    if d <= n: ### 根节点是否满足要求 (相当于c)
        rlist.append(root)
        # print(root)
    left = int(np.abs(d - n))
    right = d + n
    sub_trees = bk_tree[root] ### {1:{}, 2:{}, 3:{}}
    for dis in range(left, right+1):
        sub_tree = sub_trees.get(dis, None) ### {fame:{1:{}, 2:{}}}
        if sub_trees is not None:
            fuzzy_search(n, q, sub_tree, rlist)
            

slist = ['game', 'fame', 'gain', 'gate' ,'aim', 'same',  'gay', 'frame',  'acm', 'home']

root = slist[0]
tree = {root:{}}

for w in slist:
    if w != root:
        add_one(w, tree)

rlist = []
fuzzy_search(1, 'gan', tree, rlist)
print(rlist)