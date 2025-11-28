"""
MyRandomForestClassifier - Integrated with PA7 implementations
CPSC 322 Fall 2025
"""

import random
from collections import Counter
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myevaluation import bootstrap_sample, accuracy_score

class MyRandomForestClassifier:
    """
    Random Forest Classifier using PA7 MyDecisionTreeClassifier.
    
    Parameters:
        N: Number of decision trees to generate
        M: Number of best trees to select for the forest
        F: Number of random features to consider at each split
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, N=20, M=7, F=2, random_state=None):
        """
        Initialize random forest classifier.
        
        Args:
            N: Number of trees to generate
            M: Number of best trees to keep
            F: Number of features to consider at each split
            random_state: Random seed
        """
        self.N = N
        self.M = M
        self.F = F
        self.random_state = random_state
        self.forest = []  # Will hold M decision trees
        
    def fit(self, X_train, y_train):
        """
        Train random forest by generating N trees and selecting M best.
        
        Args:
            X_train: Training features (list of lists)
            y_train: Training labels (list)
        """
        if self.random_state is not None:
            random.seed(self.random_state)
        
        # Generate N decision trees with bootstrap sampling
        trees_with_accuracy = []
        
        for i in range(self.N):
            # Bootstrap sample
            seed = self.random_state + i if self.random_state else None
            X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(
                X_train, y_train, random_state=seed
            )
            
            # Create modified decision tree with random feature selection
            tree = MyDecisionTreeClassifierWithRandomFeatures(
                F=self.F, 
                random_state=seed
            )
            tree.fit(X_sample, y_sample)
            
            # Evaluate on out-of-bag samples
            if len(X_out_of_bag) > 0:
                predictions = tree.predict(X_out_of_bag)
                accuracy = accuracy_score(y_out_of_bag, predictions)
            else:
                # If no out-of-bag samples, use a low accuracy
                accuracy = 0.0
            
            trees_with_accuracy.append((tree, accuracy))
        
        # Select M best trees based on validation accuracy
        trees_with_accuracy.sort(key=lambda x: x[1], reverse=True)
        self.forest = [tree for tree, acc in trees_with_accuracy[:self.M]]
        
    def predict(self, X_test):
        """
        Predict classes using majority voting from M trees.
        
        Args:
            X_test: Test features (list of lists)
            
        Returns:
            List of predicted labels
        """
        # Get predictions from all trees in forest
        all_predictions = []
        for tree in self.forest:
            predictions = tree.predict(X_test)
            all_predictions.append(predictions)
        
        # Majority vote for each instance
        final_predictions = []
        for i in range(len(X_test)):
            votes = [pred[i] for pred in all_predictions]
            majority = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority)
        
        return final_predictions


class MyDecisionTreeClassifierWithRandomFeatures(MyDecisionTreeClassifier):
    """
    Extended MyDecisionTreeClassifier that supports random feature selection.
    Inherits from PA7 MyDecisionTreeClassifier.
    """
    
    def __init__(self, F=None, random_state=None):
        """
        Initialize decision tree with random feature selection.
        
        Args:
            F: Number of random features to consider at each split (None = use all)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.F = F
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fits a decision tree with random feature selection.
        
        Args:
            X_train: Training features (list of lists)
            y_train: Training labels (list)
        """
        if self.random_state is not None:
            random.seed(self.random_state)
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Determine available attributes
        n_features = len(X_train[0])
        available_attributes = list(range(n_features))
        
        # Build attribute domains
        attribute_domains = {}
        for attr_index in range(n_features):
            values = set()
            for instance in X_train:
                values.add(instance[attr_index])
            attribute_domains[attr_index] = sorted(list(values))
        
        # Build the decision tree using modified TDIDT with random feature selection
        self.tree = self._tdidt_with_random_features(
            X_train, y_train, available_attributes, attribute_domains
        )
    
    def _tdidt_with_random_features(self, instances, class_labels, available_attributes, 
                                     attribute_domains, parent_size=None):
        """
        Modified TDIDT that selects random F features at each split.
        
        Args:
            instances: Training instances
            class_labels: Class labels
            available_attributes: Available attribute indices
            attribute_domains: Dictionary of attribute domains
            parent_size: Size of parent partition
            
        Returns:
            Decision tree (nested list)
        """
        from mysklearn.myutils import (all_same_class, compute_majority_vote, 
                                       select_attribute, partition_instances)
        
        current_size = len(class_labels)
        if parent_size is None:
            parent_size = current_size
        
        # Base case 1: all same class
        if all_same_class(class_labels):
            return ["Leaf", class_labels[0], len(class_labels), parent_size]
        
        # Base case 2: no more attributes
        if len(available_attributes) == 0:
            majority_class = compute_majority_vote(class_labels)
            majority_count = class_labels.count(majority_class)
            return ["Leaf", majority_class, majority_count, parent_size]
        
        # Random feature selection: select F random features if F is specified
        if self.F is not None and self.F < len(available_attributes):
            selected_attributes = random.sample(available_attributes, self.F)
        else:
            selected_attributes = available_attributes
        
        # Select best attribute from the random subset
        split_attribute = select_attribute(instances, selected_attributes, class_labels)
        
        if split_attribute is None:
            majority_class = compute_majority_vote(class_labels)
            majority_count = class_labels.count(majority_class)
            return ["Leaf", majority_class, majority_count, parent_size]
        
        # Create tree node
        attribute_name = f"att{split_attribute}"
        tree = ["Attribute", attribute_name]
        
        # Partition instances
        partitions = partition_instances(instances, class_labels, split_attribute)
        
        # Get available values (sorted)
        available_values = sorted(attribute_domains[split_attribute])
        
        # Recursively build subtrees
        remaining_attributes = [attr for attr in available_attributes if attr != split_attribute]
        
        for value in available_values:
            value_branch = ["Value", value]
            
            if value in partitions:
                partition_instances_subset, partition_labels = partitions[value]
                subtree = self._tdidt_with_random_features(
                    partition_instances_subset, partition_labels,
                    remaining_attributes, attribute_domains, current_size
                )
                value_branch.append(subtree)
            else:
                # No instances with this value - use majority from parent
                majority_class = compute_majority_vote(class_labels)
                majority_count = class_labels.count(majority_class)
                value_branch.append(["Leaf", majority_class, majority_count, current_size])
            
            tree.append(value_branch)
        
        return tree


if __name__ == "__main__":
    print("MyRandomForestClassifier (PA7 Integration) loaded successfully")
