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

if __name__ == "__main__":
    # test = np.array([[1, 2], [2, 1], [3, 4], [5, 4], [6, 2]])
    test = np.array([[0, 1], [1, 0], [2, 3], [3, 0], [5, 0], [6, 0]])
    test += 10
    test = [[50, 51], [51, 50], [52, 50], [53, 50], [54, 50], [55, 50], [58, 50], [59, 50], [61, 61], [62, 50], [63, 50], [66, 50], [67, 50], [72, 50], [73, 50], [75, 75], [77, 50]]
    print(process_propagation(test))