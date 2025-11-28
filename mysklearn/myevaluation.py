from mysklearn import myutils
import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
        test_size(float or int): float for proportion or int for absolute number
        random_state(int): integer used for seeding a random number generator
        shuffle(bool): whether or not to randomize the order before splitting

    Returns:
        X_train, X_test, y_train, y_test
    """
    n_samples = len(X)

    indices = list(range(n_samples))
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = test_size

    if shuffle:
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        n_train = n_samples - n_test
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = list(range(n_samples))
    
    if shuffle:
        np.random.shuffle(indices)

    fold_size = n_samples // n_splits
    remainder = n_samples % n_splits
    
    folds = []
    start_idx = 0
    
    for i in range(n_splits):
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_fold_size

        test_indices = indices[start_idx:end_idx]

        train_indices = indices[:start_idx] + indices[end_idx:]
        
        folds.append((train_indices, test_indices))
        start_idx = end_idx
    
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
        y(list of obj): The target y values (parallel to X).
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator
        shuffle(bool): whether or not to randomize the order before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds
    """
    n_samples = len(X)

    if shuffle:
        indexed_labels = list(enumerate(y))
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indexed_labels)
        else:
            np.random.shuffle(indexed_labels)

        class_indices = {}
        for idx, label in indexed_labels:
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
    else:
        class_indices = {}
        for i, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

    folds = [[] for _ in range(n_splits)]
    
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]

        for i, idx in enumerate(indices):
            fold_num = i % n_splits
            folds[fold_num].append(idx)

    result_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = []
        for j in range(n_splits):
            if j != i:
                train_indices.extend(folds[j])
        result_folds.append((train_indices, test_indices))
    
    return result_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
        n_samples(int): Number of samples to generate
        random_state(int): integer used for seeding a random number generator

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_samples is None:
        n_samples = len(X)

    sampled_indices = np.random.choice(len(X), size=n_samples, replace=True)

    all_indices = set(range(len(X)))
    sampled_set = set(sampled_indices)
    out_of_bag_indices = list(all_indices - sampled_set)
    
    X_sample = [X[i] for i in sampled_indices]
    X_out_of_bag = [X[i] for i in out_of_bag_indices]
    
    if y is not None:
        y_sample = [y[i] for i in sampled_indices]
        y_out_of_bag = [y[i] for i in out_of_bag_indices]
    else:
        y_sample = None
        y_out_of_bag = None
    
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
        y_pred(list of obj): The predicted target y values (parallel to y_true)
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    """
    n_labels = len(labels)
    matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = labels.index(true_label)
        pred_index = labels.index(pred_label)
        matrix[true_index][pred_index] += 1
    
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
        y_pred(list of obj): The predicted target y values (parallel to y_true)
        normalize(bool): If False, return the number of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples,
            else returns the number of correctly classified samples.
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    
    if normalize:
        return correct / len(y_true)
    else:
        return correct

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp += 1
            else:
                fp += 1

    if (tp + fp) == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    
    if pos_label is None:
        pos_label = labels[0]

    tp = 0 
    fn = 0 
    
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            if y_pred[i] == pos_label:
                tp += 1
            else:
                fn += 1

    if (tp + fn) == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if (precision + recall) == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1