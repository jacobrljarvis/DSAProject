from mysklearn import myutils
import numpy as np
from collections import Counter


class MyKNeighborsClassifier:
    """k-Nearest Neighbors classifier.
    
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of obj): The list of training instances
        y_train(list of obj): The target y values
    """
    
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        
        Args:
            X_train(list of list of obj): The list of training instances
            y_train(list of obj): The target y values
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        
        Args:
            X_test(list of list of obj): The list of testing samples
            
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
            neighbor_indices(list of list of int): 2D list of k nearest neighbor indices
        """
        distances = []
        neighbor_indices = []
        
        for test_instance in X_test:
            # Calculate distances to all training instances
            row_distances = []
            for i, train_instance in enumerate(self.X_train):
                dist = self._euclidean_distance(test_instance, train_instance)
                row_distances.append((dist, i))
            
            # Sort by distance and get k nearest
            row_distances.sort(key=lambda x: x[0])
            k_nearest = row_distances[:self.n_neighbors]
            
            # Extract distances and indices
            dists = [d for d, _ in k_nearest]
            indices = [i for _, i in k_nearest]
            
            distances.append(dists)
            neighbor_indices.append(indices)
        
        return distances, neighbor_indices
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        
        Args:
            X_test(list of list of obj): The list of testing samples
            
        Returns:
            y_predicted(list of obj): The predicted target y values
        """
        predictions = []
        
        for test_instance in X_test:
            # Calculate distances to all training instances
            distances = []
            for i, train_instance in enumerate(self.X_train):
                dist = self._euclidean_distance(test_instance, train_instance)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.n_neighbors]
            
            # Majority vote
            k_labels = [label for _, label in k_nearest]
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        
        return predictions
    
    def _euclidean_distance(self, instance1, instance2):
        """Calculate Euclidean distance between two instances.
        
        Args:
            instance1(list): First instance
            instance2(list): Second instance
            
        Returns:
            float: Euclidean distance
        """
        return np.sqrt(sum((float(a) - float(b)) ** 2 
                          for a, b in zip(instance1, instance2)))


class MyNaiveBayesClassifier:
    """Gaussian Naive Bayes classifier.
    
    Attributes:
        priors(dict): The class prior probabilities computed from the training set
        posteriors(dict): The class conditional probabilities (Gaussian parameters)
    """
    
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier."""
        self.priors = None
        self.posteriors = None
        self.classes = None
    
    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        
        Args:
            X_train(list of list of obj): The list of training instances
            y_train(list of obj): The target y values
        """
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train)
        
        self.classes = np.unique(y_train)
        n_samples = len(y_train)
        n_features = X_train.shape[1]
        
        # Calculate priors and posteriors for each class
        self.priors = {}
        self.posteriors = {}
        
        for c in self.classes:
            # Get instances of this class
            X_c = X_train[y_train == c]
            
            # Prior probability
            self.priors[c] = len(X_c) / n_samples
            
            # Posterior parameters (mean and std for each feature)
            self.posteriors[c] = {}
            for feature_idx in range(n_features):
                feature_values = X_c[:, feature_idx]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                if std == 0:  # Avoid division by zero
                    std = 1e-6
                self.posteriors[c][feature_idx] = (mean, std)
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        
        Args:
            X_test(list of list of obj): The list of testing samples
            
        Returns:
            y_predicted(list of obj): The predicted target y values
        """
        X_test = np.array(X_test, dtype=float)
        predictions = []
        
        for instance in X_test:
            # Calculate posterior for each class
            class_posteriors = {}
            
            for c in self.classes:
                # Start with log prior
                posterior = np.log(self.priors[c])
                
                # Add log likelihoods for each feature
                for feature_idx, value in enumerate(instance):
                    mean, std = self.posteriors[c][feature_idx]
                    likelihood = self._gaussian_probability(value, mean, std)
                    posterior += np.log(likelihood + 1e-10)  # Avoid log(0)
                
                class_posteriors[c] = posterior
            
            # Predict class with highest posterior
            prediction = max(class_posteriors, key=class_posteriors.get)
            predictions.append(prediction)
        
        return predictions
    
    def _gaussian_probability(self, x, mean, std):
        """Calculate Gaussian probability density.
        
        Args:
            x(float): Value
            mean(float): Mean of distribution
            std(float): Standard deviation
            
        Returns:
            float: Probability density
        """
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)


class MyDummyClassifier:
    """Dummy classifier that always predicts the most frequent class.
    
    Attributes:
        most_common_label(obj): The most frequent class label from training
    """
    
    def __init__(self):
        """Initializer for MyDummyClassifier."""
        self.most_common_label = None
    
    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        
        Args:
            X_train(list of list of obj): The list of training instances (not used)
            y_train(list of obj): The target y values
        """
        # Find most common class
        self.most_common_label = Counter(y_train).most_common(1)[0][0]
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        
        Args:
            X_test(list of list of obj): The list of testing samples
            
        Returns:
            y_predicted(list of obj): The predicted target y values (all same)
        """
        return [self.most_common_label] * len(X_test)


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Determine available attributes (all attributes by index)
        n_features = len(X_train[0])
        available_attributes = list(range(n_features))
        
        # Build attribute domains (sorted values for each attribute)
        attribute_domains = {}
        for attr_index in range(n_features):
            values = set()
            for instance in X_train:
                values.add(instance[attr_index])
            attribute_domains[attr_index] = sorted(list(values))
        
        # Build the decision tree using TDIDT
        self.tree = myutils.tdidt(X_train, y_train, available_attributes, attribute_domains)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = myutils.predict_with_tree(self.tree, instance)
            predictions.append(prediction)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        def extract_rules(tree, rule_conditions):
            """Recursively extract rules from tree."""
            if tree[0] == "Leaf":
                # Base case: we've reached a leaf, print the rule
                class_label = tree[1]
                rule_str = "IF " + " AND ".join(rule_conditions) + f" THEN {class_name} = {class_label}"
                print(rule_str)
            elif tree[0] == "Attribute":
                # Get attribute name
                attr_name = tree[1]
                # Convert to custom name if provided
                if attribute_names is not None and attr_name.startswith("att"):
                    attr_index = int(attr_name[3:])
                    attr_name = attribute_names[attr_index]
                
                # Recurse on each value branch
                for i in range(2, len(tree)):
                    value_branch = tree[i]
                    if value_branch[0] == "Value":
                        value = value_branch[1]
                        new_condition = f"{attr_name} == {value}"
                        extract_rules(value_branch[2], rule_conditions + [new_condition])
        
        if self.tree is not None:
            extract_rules(self.tree, [])

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
