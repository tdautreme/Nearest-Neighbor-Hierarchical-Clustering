import numpy as np

def process_propagation(propagation_array):
    def find_root(label, parent):
        if parent[label] == label:
            return label
        parent[label] = find_root(parent[label], parent)
        return parent[label]

    def union_labels(label1, label2, parent):
        root1 = find_root(label1, parent)
        root2 = find_root(label2, parent)
        if root1 != root2:
            parent[root1] = root2


    label_to_root = {}
    for src, dst in propagation_array:
        if src not in label_to_root:
            label_to_root[src] = src
        if dst not in label_to_root:
            label_to_root[dst] = dst

    for src, dst in propagation_array:
        union_labels(src, dst, label_to_root)

    root_to_labels = {}
    for label in label_to_root:
        root = find_root(label, label_to_root)
        if root not in root_to_labels:
            root_to_labels[root] = []
        root_to_labels[root].append(label)

    result_array = []
    for labels in root_to_labels.values():
        if len(labels) > 1:
            # Create a sub-array with old and new labels
            sub_array = np.array([[old_label, labels[0]] for old_label in labels])
            result_array.append(sub_array)
    # reshape to make (n, 2) array
    result_array = np.concatenate(result_array)
    return result_array

class Node:
    def __init__(self, label):
        self.label = label
        self.reduced_target = None
        self.eated = False

    def set_intial_target(self, target):
        self.initial_target = target

    def get_target(self):
        if self.reduced_target != None:
            return self.reduced_target
        return self.initial_target
    
    def search_main_target(self):
        current_target = self.get_target()
        while current_target.eated:
            current_target = current_target.get_target()
        return current_target

    def go_in_target(self):
        self.reduced_target = self.search_main_target()
        if self.reduced_target == None or self.reduced_target == self:
            return 0
        self.eated = True
        return 1
        
def union_find(propagation_array):
    labels = propagation_array[:, 0]
    target_labels = propagation_array[:, 1]
    target_labels_indexes_in_labels = np.searchsorted(labels, target_labels)
    # create node foreach col_1 in np array
    nodes = [Node(label) for label in labels]
    # set initial target foreach node
    for i, node in enumerate(nodes):
        target_node = nodes[target_labels_indexes_in_labels[i]]
        node.set_intial_target(target_node)

    for node in nodes:
        node.go_in_target()
    return np.array([[node.label, node.reduced_target.label] for node in nodes])

if __name__ == "__main__":
    # test_1 = [[50, 51], [51, 50], [52, 50], [53, 50], [54, 50], [55, 50], [58, 50], [59, 50], [61, 61], [62, 50], [63, 50], [66, 50], [67, 50], [72, 50], [73, 50], [75, 75], [77, 50]]
    
    
    # print(process_propagation(test_1))

    # distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    test_2 = np.array([[1, 2], [2, 1], [3, 4], [4, 5], [5, 3], [6, 1]])
    # result should be [[1, 2], [2, 2], [3, 5], [4, 5], [5, 5], [6, 2]] 
    print(union_find(test_2))
    # print(process_propagation(test_2))



