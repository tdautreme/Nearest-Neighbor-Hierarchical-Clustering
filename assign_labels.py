import numpy as np
from utils import timeit
from nnhc import get_labels

@timeit
def assign_labels_process_propagation(label_propagation_array, y, labels, min_label_count=2):
    # add offset to propagation_order_array
    label_offset = np.max(y) + 1
    label_propagation_array[:, 1] += label_offset

    labels_count = labels.shape[0]
    new_diffrent_labels = []
    # foreach label propagation, update the label in x
    for source_label, target_label in label_propagation_array:
        y[y == source_label] = target_label
        labels_count -= 1
        if target_label not in new_diffrent_labels:
            new_diffrent_labels.append(target_label)

        # If we have enough labels, stop
        if labels_count + len(new_diffrent_labels) == min_label_count:
            break
    # y, labels = normalize_labels(y)
    labels = get_labels(y)
    return y, labels

# @timeit
# def assign_labels_union_find(label_propagation_array, y, labels, min_label_count=2):
#     for source_label, target_label in label_propagation_array:
#         if source_label == target_label:
#             continue
#         y[y == source_label] = target_label
#         current_label_count -= 1
#         if current_label_count <= min_label_count:
#             break
#     labels = get_labels(y)	
#     return y, labels

@timeit
def assign_labels_union_find(label_propagation_array, y, labels, min_label_count=2):
    # remove row that have same source and target
    label_propagation_array = label_propagation_array[label_propagation_array[:, 0] != label_propagation_array[:, 1]]
    
    # compute max row count using min_label_count
    labels_count = labels.shape[0]
    max_row_count = labels_count - min_label_count
    # get max_row_count first rows
    label_propagation_array = label_propagation_array[:max_row_count]

    for source_label, target_label in label_propagation_array:
        y[y == source_label] = target_label

    return y, get_labels(y)