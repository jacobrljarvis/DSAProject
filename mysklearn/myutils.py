# TODO: your reusable general-purpose functions here
import numpy as np
import math
from copy import deepcopy

def compute_entropy(class_labels):
    """Compute entropy for a list of class labels.
    
    Args:
        class_labels(list): List of class labels
        
    Returns:
        float: Entropy value
    """
    if len(class_labels) == 0:
        return 0
    
    # Count occurrences of each class
    class_counts = {}
    for label in class_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate entropy
    entropy = 0
    total = len(class_labels)
    for count in class_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy

def compute_partition_entropy(partitions):
    """Compute weighted entropy for a set of partitions.
    
    Args:
        partitions(dict): Dictionary mapping attribute values to lists of class labels
        
    Returns:
        float: Weighted entropy value
    """
    total_instances = sum(len(labels) for labels in partitions.values())
    if total_instances == 0:
        return 0
    
    weighted_entropy = 0
    for labels in partitions.values():
        if len(labels) > 0:
            weight = len(labels) / total_instances
            weighted_entropy += weight * compute_entropy(labels)
    
    return weighted_entropy

def select_attribute(instances, attributes, class_labels):
    """Select the best attribute to partition on using entropy.
    
    Args:
        instances(list of list): Training instances
        attributes(list of int): Available attribute indices
        class_labels(list): Class labels parallel to instances
        
    Returns:
        int: Index of the best attribute to split on
    """
    if len(attributes) == 0:
        return None
    
    min_entropy = float('inf')
    best_attribute = attributes[0]
    
    for attr_index in attributes:
        # Create partitions based on this attribute
        partitions = {}
        for i, instance in enumerate(instances):
            attr_value = instance[attr_index]
            if attr_value not in partitions:
                partitions[attr_value] = []
            partitions[attr_value].append(class_labels[i])
        
        # Calculate weighted entropy for this attribute
        entropy = compute_partition_entropy(partitions)
        
        if entropy < min_entropy:
            min_entropy = entropy
            best_attribute = attr_index
    
    return best_attribute

def partition_instances(instances, class_labels, split_attribute):
    """Partition instances based on a split attribute.
    
    Args:
        instances(list of list): Training instances
        class_labels(list): Class labels parallel to instances
        split_attribute(int): Index of attribute to split on
        
    Returns:
        dict: Dictionary mapping attribute values to (instances, labels) tuples
    """
    partitions = {}
    
    for i, instance in enumerate(instances):
        attr_value = instance[split_attribute]
        if attr_value not in partitions:
            partitions[attr_value] = ([], [])
        partitions[attr_value][0].append(instance)
        partitions[attr_value][1].append(class_labels[i])
    
    return partitions

def compute_majority_vote(class_labels):
    """Compute majority class from a list of class labels.
    
    Args:
        class_labels(list): List of class labels
        
    Returns:
        str: Majority class label (alphabetically first in case of tie)
    """
    if len(class_labels) == 0:
        return None
    
    # Count occurrences
    class_counts = {}
    for label in class_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Find max count
    max_count = max(class_counts.values())
    
    # Get all classes with max count (for handling ties)
    majority_classes = [label for label, count in class_counts.items() if count == max_count]
    
    # Return alphabetically first in case of tie
    return sorted(majority_classes)[0]

def all_same_class(class_labels):
    """Check if all class labels are the same.
    
    Args:
        class_labels(list): List of class labels
        
    Returns:
        bool: True if all labels are the same
    """
    if len(class_labels) == 0:
        return True
    return len(set(class_labels)) == 1

def tdidt(instances, class_labels, available_attributes, attribute_domains, header=None, parent_size=None):
    """TDIDT (Top-Down Induction of Decision Tree) algorithm.
    
    Args:
        instances(list of list): Training instances
        class_labels(list): Class labels parallel to instances
        available_attributes(list of int): Available attribute indices
        attribute_domains(dict): Dictionary mapping attribute indices to their possible values
        header(list of str): Optional header for attribute names
        parent_size(int): Total number of instances from parent partition
        
    Returns:
        tree: Nested list representation of decision tree
    """
    # Track the total size at this level
    current_size = len(class_labels)
    if parent_size is None:
        parent_size = current_size
    
    # Base case 1: all instances have same class
    if all_same_class(class_labels):
        return ["Leaf", class_labels[0], len(class_labels), parent_size]
    
    # Base case 2: no more attributes to partition on (clash)
    if len(available_attributes) == 0:
        majority_class = compute_majority_vote(class_labels)
        majority_count = class_labels.count(majority_class)
        return ["Leaf", majority_class, majority_count, parent_size]
    
    # Select attribute with lowest entropy
    split_attribute = select_attribute(instances, available_attributes, class_labels)
    
    # Get attribute name
    if header is not None:
        attribute_name = header[split_attribute]
    else:
        attribute_name = f"att{split_attribute}"
    
    # Create tree node
    tree = ["Attribute", attribute_name]
    
    # Partition instances based on split attribute
    partitions = partition_instances(instances, class_labels, split_attribute)
    
    # Get available values for this attribute (sorted alphabetically)
    available_values = sorted(attribute_domains[split_attribute])
    
    # Recursively build subtrees for each attribute value
    remaining_attributes = [attr for attr in available_attributes if attr != split_attribute]
    
    for value in available_values:
        # Create value branch
        value_branch = ["Value", value]
        
        if value in partitions:
            # Recursively build subtree
            partition_instances_subset, partition_labels = partitions[value]
            # Pass the current partition size as parent_size for the next level
            subtree = tdidt(partition_instances_subset, partition_labels, 
                          remaining_attributes, attribute_domains, header, current_size)
            value_branch.append(subtree)
        else:
            # No instances with this value - use majority vote from parent
            majority_class = compute_majority_vote(class_labels)
            majority_count = class_labels.count(majority_class)
            value_branch.append(["Leaf", majority_class, majority_count, current_size])
        
        tree.append(value_branch)
    
    return tree

def predict_with_tree(tree, instance):
    """Make a prediction for an instance using a decision tree.
    
    Args:
        tree: Nested list representation of decision tree
        instance(list): Instance to predict
        
    Returns:
        str: Predicted class label
    """
    if tree[0] == "Leaf":
        return tree[1]
    
    # Get attribute to split on
    attribute_name = tree[1]
    # Extract attribute index from name (e.g., "att0" -> 0)
    if attribute_name.startswith("att"):
        attr_index = int(attribute_name[3:])
    else:
        # This shouldn't happen with our default naming, but handle it
        # Return majority class from all leaf nodes in this subtree
        return _get_majority_from_subtree(tree)
    
    # Get the value of this attribute in the instance
    instance_value = instance[attr_index]
    
    # Find the matching value branch
    for i in range(2, len(tree)):
        value_branch = tree[i]
        if value_branch[0] == "Value" and value_branch[1] == instance_value:
            # Recursively predict with subtree
            return predict_with_tree(value_branch[2], instance)
    
    # If no matching branch found (unseen value), return majority class from this subtree
    return _get_majority_from_subtree(tree)


def _get_majority_from_subtree(tree):
    """Extract majority class from all leaf nodes in a subtree.
    
    Args:
        tree: Nested list representation of decision tree
        
    Returns:
        str: Majority class label
    """
    from collections import Counter
    
    def collect_leaf_classes(node):
        """Recursively collect all class labels from leaf nodes."""
        classes = []
        
        if not node or len(node) == 0:
            return classes
            
        node_type = node[0]
        
        if node_type == "Leaf":
            # Leaf node: ["Leaf", class_label, count, total]
            classes.append(node[1])
        elif node_type == "Attribute":
            # Attribute node: ["Attribute", attr_name, value_branch1, value_branch2, ...]
            # Recursively check all value branches (starting at index 2)
            for i in range(2, len(node)):
                value_branch = node[i]
                classes.extend(collect_leaf_classes(value_branch))
        elif node_type == "Value":
            # Value node: ["Value", value, subtree]
            # Recursively check the subtree (at index 2)
            if len(node) > 2:
                classes.extend(collect_leaf_classes(node[2]))
        
        return classes
    
    all_classes = collect_leaf_classes(tree)
    if all_classes:
        return Counter(all_classes).most_common(1)[0][0]
    
    # If we somehow get no classes, return a default
    # This shouldn't happen but prevents None returns
    return "C"  # Default to most common position
