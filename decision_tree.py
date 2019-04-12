import numpy as np
import os
from collections import defaultdict
import math
import graphviz

def partition(x):
    x_subsets = defaultdict(list)
    for x1 in range(len(x)): 
        x_subsets[x[x1]].append(x1)
    return x_subsets

def entropy(y):

    counts = defaultdict(int)
    total=0
    entropy=0
    for x in range(len(y)): 
        counts[y[x]]=int(counts[y[x]])+1
    for key, value in counts.items():
        total+=value
    for key, value in counts.items():
        entropy+= (value/total)*math.log(value/total,2)
    return entropy*-1

def mutual_information(x, y):

    y_entropy=entropy(y)
    MI = defaultdict(int)
    indices=np.unique(x)
    for i in indices:
        cond_entropy=0
        t=defaultdict(list)
        for j in range (len(x)):
            if x[j] == i:
                t[0].append(y[j])
            else:
                t[1].append(y[j])
        for key, value in t.items():
            cond_entropy+=(len(value)/len(x))*entropy(value)
        MI[i]=y_entropy-cond_entropy
    return MI  


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    decision_tree=dict()
    vector=0
    vector_value=0
    if(attribute_value_pairs==None):
        attribute_value_pairs=[]
        for i in range(len(x[0])):
            unique_list=[]
            for l in x[:,i]:  
                if l not in unique_list:
                    unique_list.append(l)
                    attribute_value_pairs.append(tuple((i,l)))
    list_vectors=[]
    if (depth==max_depth  or len(attribute_value_pairs)==0):
        return (np.argmax(np.bincount(y)))
    if(all(p == y[0] for p in y)):
        return y[0]
    for i in attribute_value_pairs:
        list_vectors.append(i[0])
    uniq_list=np.unique(list_vectors)
    mi=defaultdict(int)
    max=0
    for i in uniq_list:
        mi=mutual_information(x[:,i],y)
        for key , value  in mi.items():
            if((i,key) in attribute_value_pairs):
                if max < value:
                    max=value
                    vector=i
                    vector_value=key
    attribute_value_pairs.remove((vector,vector_value))
    x_left=[]
    x_right=[]
    y_left=[]
    y_right=[]
    for k ,p in zip(x,y):
        if(k[vector]==vector_value):
            x_left.append(k)
            y_left.append(p)
        else:
            x_right.append(k)
            y_right.append(p)
    array_copy=attribute_value_pairs.copy()
    decision_tree[vector,vector_value,True]=id3(np.asarray(x_left),y_left,array_copy,depth+1,max_depth)
    array_copy=attribute_value_pairs.copy()
    decision_tree[vector,vector_value,False]=id3(np.asarray(x_right),y_right,array_copy,depth+1,max_depth)
    return decision_tree


def predict_example(x, tree):
    if type(tree) is not dict:
        return tree
    for key in tree.keys():
        if((x[key[0]]==key[1])):
            return(predict_example(x,tree[key[0],key[1],True]))
        else:
            return(predict_example(x,tree[key[0],key[1],False]))

def compute_error(y_true, y_pred):
    count=0
    for i in range(len(y_true)):
        if y_true[i]!=y_pred[i]:
            count = count+1
    return (count/len(y_pred))

def pretty_print(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):

    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    decision_tree = id3(Xtrn, ytrn, max_depth=5)
    pretty_print(decision_tree)
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
