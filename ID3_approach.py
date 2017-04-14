import queue
import math
from collections import Counter
from time import sleep

import gc
import pandas as pd
from copy import deepcopy

from treelib import Node, Tree


# ref: https://gist.github.com/whitehaven/bbd408edca38de93637635b52d2bba89


def ID3_entropies(data_df):
    """
    Takes pandas.DataFrame and returns a series with all non-index schemas' entropies calculated.

    It supports non-binary field types by calculating average entropy. Result series starts with the most productive decision level.
    """

    def entropy_for_field(field):
        entropy = 0
        field_entry_count = len(field)

        # get count of unique
        field_counter = Counter(field)

        # E( Si/S * E(pi*log2(pi)) )
        for trait, count in field_counter.items():
            p_T = count / field_entry_count
            p_F = (field_entry_count - count) / field_entry_count

            if p_T == 0 or p_F == 0:
                entropy = 0
                break
            # Si/S * E(pi*log2(pi))
            entropy += count / field_entry_count * (- (p_T * math.log2(p_T)) - (p_F * math.log2(p_F)))
        return entropy

    data_df_entropy = {}
    for field in data_df:
        if(field!='output'):
            entropy_this_field = entropy_for_field(data_df[field])
            data_df_entropy[field] = entropy_this_field


    data_df_entropy_se = pd.Series(data_df_entropy)


    data_df_entropy_se.sort_values(ascending=False, inplace=True)
    return data_df_entropy_se




"""
with open("./data/trainingset.txt", "rt") as fin:
    with open("./data/trainingset_parsed.txt", "wt") as fout:
        for line in fin:
            line = line.replace('\"?\"', '\"\"')
            fout.write(line)
fin.close()
fout.close()
training = pd.read_csv("./data/trainingset_parsed.txt", header=None)
training.columns = ['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact'
    , 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'output']
entropy = ID3_entropies(training)

# Parsing this data for the first time
training.drop('id', axis=1, inplace=True)
# training.drop('day', axis=1, inplace=True)
# training.drop('month', axis=1, inplace=True)
# training.drop('contact', axis=1, inplace=True)
for index, single_line in entropy.iteritems():
    print(index, single_line)
    if (single_line == 0.0):  # Filter the ones with 0 entropy
        training.drop(index, axis=1, inplace=True)

# Leveling different continue features into categorized
age_arr = [-10, 31, 37, 48]
balance = [-10, 75, 451, 1420]
last_contact = [-10, 8, 16, 21] #is dropped
campaign = [-10, 1, 2, 3]
pdays = [-10, -1, 100, 300]
previous = [-10, 0, 10, 30]

for index,inline in training.iterrows():

    temp_val =inline['age']
    temp_index = 0
    for i in range(1, age_arr.__len__()):
        if(temp_val <= age_arr[i] and temp_val >= age_arr[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = age_arr.__len__()
    training.loc[index,"age"] = temp_index

    temp_val =inline['balance']
    temp_index = 0
    for i in range(1, balance.__len__()):
        if(temp_val <= balance[i] and temp_val >= balance[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = balance.__len__()
    training.loc[index,"balance"] = temp_index

    temp_val =inline['campaign']
    temp_index = 0
    for i in range(1, campaign.__len__()):
        if(temp_val <= campaign[i] and temp_val >= campaign[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = campaign.__len__()
    training.loc[index,"campaign"] = temp_index

    temp_val =inline['pdays']
    temp_index = 0
    for i in range(1, pdays.__len__()):
        if(temp_val <= pdays[i] and temp_val >= pdays[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = pdays.__len__()
    training.loc[index,"pdays"] = temp_index

    temp_val =inline['previous']
    temp_index = 0
    for i in range(1, previous.__len__()):
        if(temp_val <= previous[i] and temp_val >= previous[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = previous.__len__()
    training.loc[index,"previous"] = temp_index

training.to_csv('./data/training_2_2.txt', index=False)


print(training)
"""




current_node_index = 0
gc_counting = 0

class NodeData(object):
    def __init__(self, parent_id, current_f, value, feature_entropy, data_set):
        self.current_f = current_f
        self.dataset = data_set
        self.children = []
        self.parent_id = parent_id
        self.value = value
        self.feature_entropy = feature_entropy
        self.result = ""
        self.next_feature = ""

    def __cmp__(self, other):
        return 1

def iterate_depth_first_generation(c_node):
    if(len(c_node.data.feature_entropy) > 1 and len(c_node.data.dataset) > 0 ):
        features_by_order = ID3_entropies( c_node.data.dataset )
        # c_node.data.current_f = features_by_order.index[0]
        field_counter = Counter(c_node.data.dataset[features_by_order.index[0]])
        c_node.data.next_feature = features_by_order.index[0]

        # Split into different nodes if it is more than 1
        if(len(field_counter) > 1):
            type_counter = Counter(c_node.data.dataset['output'])
            typeA_num = type_counter['TypeA']
            typeB_num = type_counter['TypeB']
            if (typeA_num >= typeB_num):
                c_node.data.result = 0
            else:
                c_node.data.result = 1
            print(c_node.data.result)
            for index, val in field_counter.items():
                global  current_node_index
                current_node_index = current_node_index + 1
                if(str(index).isnumeric()):
                    new_dataset = c_node.data.dataset.query(features_by_order.index[0]+'=='+str(index))
                else:
                    new_dataset = c_node.data.dataset.query(features_by_order.index[0]+'==\''+str(index)+'\'')
                new_dataset = new_dataset.drop(features_by_order.index[0], axis=1)
                new_entropy = features_by_order.drop(features_by_order.index[0])
                node_to_be_added = NodeData(c_node.identifier, features_by_order.index[0], index, new_entropy, new_dataset)
                label_name = features_by_order.index[0]+":"+str(index)

                temp_node = tree.create_node(label_name, current_node_index, c_node.identifier, data=node_to_be_added)
                iterate_depth_first_generation(temp_node)
        else:
            # Comment here is used for tracking expected output
            # cc_node = c_node
            # condition_arr = {}
            # while 1:
            #     if (cc_node.data.parent_id != ""):
            #         condition_arr[cc_node.data.current_f] = cc_node.data.value
            #         cc_node = tree.get_node(cc_node.data.parent_id)
            #     else:
            #         break
            # print(condition_arr)

            # Calculate the majority
            type_counter = Counter(c_node.data.dataset['output'])
            typeA_num = type_counter['TypeA']
            typeB_num = type_counter['TypeB']
            if(typeA_num >= typeB_num):
                c_node.data.result = 0
            else:
                c_node.data.result = 1
            print(c_node.data.result)

    else:
        # del c_node.data.dataset
        # memory management unit
        global gc_counting
        gc_counting = gc_counting + 1
        if gc_counting > 1000:
            gc.collect ()
            gc_counting = 0
        # Determine under this condition which output should be given
            # print(a,b)
    del c_node.data.feature_entropy
    # tree.show()
    # print(current_node_index)

    return 1

training = pd.read_csv("./data/training_2_2.txt")
tree = Tree()
entropy = ID3_entropies(training)
current_node = tree.create_node("root", 0, data=NodeData("","", "", entropy, training) )

iterate_depth_first_generation( current_node )

# tree.show()

root = tree.get_node(0)
print(root.data.next_feature)



"""
# Parsing query data

with open("./data/queries.txt", "rt") as fin:
    with open("./data/new_queries_2.txt", "wt") as fout:
        for line in fin:
            line = line.replace('\"?\"', '\"\"')
            fout.write(line)
fin.close()
fout.close()
query = pd.read_csv("./data/new_queries_2.txt", header=None)
query.columns = ['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact'
    , 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'output']
entropy = ID3_entropies(query)

# Parsing this data for the first time
# query.drop('id', axis=1, inplace=True)
query.drop('output', axis=1, inplace=True)
# training.drop('day', axis=1, inplace=True)
# training.drop('month', axis=1, inplace=True)
# training.drop('contact', axis=1, inplace=True)
for index, single_line in entropy.iteritems():
    print(index, single_line)
    if (single_line == 0.0):  # Filter the ones with 0 entropy
        query.drop(index, axis=1, inplace=True)

# Leveling different continue features into categorized
age_arr = [-10, 31, 37, 48]
balance = [-10, 75, 451, 1420]
last_contact = [-10, 8, 16, 21] #is dropped
campaign = [-10, 1, 2, 3]
pdays = [-10, -1, 100, 300]
previous = [-10, 0, 10, 30]

for index,inline in query.iterrows():

    temp_val =inline['age']
    temp_index = 0
    for i in range(1, age_arr.__len__()):
        if(temp_val <= age_arr[i] and temp_val >= age_arr[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = age_arr.__len__()
    query.loc[index,"age"] = temp_index

    temp_val =inline['balance']
    temp_index = 0
    for i in range(1, balance.__len__()):
        if(temp_val <= balance[i] and temp_val >= balance[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = balance.__len__()
    query.loc[index,"balance"] = temp_index

    temp_val =inline['campaign']
    temp_index = 0
    for i in range(1, campaign.__len__()):
        if(temp_val <= campaign[i] and temp_val >= campaign[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = campaign.__len__()
    query.loc[index,"campaign"] = temp_index

    temp_val =inline['pdays']
    temp_index = 0
    for i in range(1, pdays.__len__()):
        if(temp_val <= pdays[i] and temp_val >= pdays[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = pdays.__len__()
    query.loc[index,"pdays"] = temp_index

    temp_val =inline['previous']
    temp_index = 0
    for i in range(1, previous.__len__()):
        if(temp_val <= previous[i] and temp_val >= previous[i-1]):
            temp_index = i
    if temp_index == 0:
        temp_index = previous.__len__()
    query.loc[index,"previous"] = temp_index

query.to_csv('./data/new_queries_2.txt', index=False)


print(query)

"""


query = pd.read_csv('./data/new_queries_2.txt')
for index, row in query.iterrows():
    temp_root = root
    while 1:
        c_nodes = tree.children(temp_root.identifier)
        have_hit = False
        if(len(c_nodes) > 0):
            for single_node in c_nodes:
                if(single_node.data.value == row[single_node.data.current_f]):
                    have_hit = single_node.identifier
        else:
            if(temp_root.data.result == 0 ):
                query.loc[index,'output'] = "TypeA"
            else:
                query.loc[index,'output'] = "TypeB"
            break
        if(have_hit!=False):
            temp_root = tree.get_node(have_hit)
        else:
            if (temp_root.data.result == 0):
                query.loc[index, 'output'] = "TypeA"
            else:
                query.loc[index, 'output'] = "TypeB"
            break

cols_to_keep = ['id', 'output']
query[cols_to_keep].to_csv('./data/output.txt', index=False)

