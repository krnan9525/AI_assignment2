import queue

from treelib import Node, Tree


current_leaves = queue.PriorityQueue()
class NodeData(object):
        def __init__(self, name, reminder):
            self.name = name
            self.reminder = reminder

        def __cmp__(self, other):
            return 1

current_leaves.put(NodeData(1,1))
print(current_leaves)
tree = Tree()
current_node = tree.create_node("hi",0,data = NodeData("hi","hkljsdf"))
print(current_node.data.reminder)
tree.show()
