import json

class TreeNode(object):
    def __init__(self, labelname, level):
        self.labelname = labelname
        self.level = level
        self.child = []

    def add_child(self, child):
        self.child.append(child)

def build_tree(jsonobj, level=0):
    node = TreeNode(jsonobj['LabelName'], level)
    for child in jsonobj.get('Subcategory', []):
        childnode = build_tree(child, level+1)
        node.add_child(childnode)
    return node

def get_all_nodes(rootnode):
    xlist = []
    xlist.append(rootnode)
    for node in rootnode.child:
        xlist.extend(get_all_nodes(node))
    return xlist

def get_node_hierarchy(node):
    if len(node.child) == 0:
        return 0
    mlist = []
    for child in node.child:
        mlist.append(get_node_hierarchy(child))
    return max(mlist) + 1

def get_2ndlevel_class(class_hierarchy_json):
    with open(class_hierarchy_json) as f:
        class_hierarchy = json.load(f)

    root = build_tree(class_hierarchy)

    mlist = list()
    for node in get_all_nodes(root):
        if get_node_hierarchy(node) == 1:
            mlist.append(node.labelname)
    return mlist