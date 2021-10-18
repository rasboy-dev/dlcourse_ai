def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    f1 = 0
    accuracy = 0

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    num_samples = prediction.shape[0]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(num_samples):
        if ground_truth[i]:
            if prediction[i]:
                tp += 1
            else:
                fn += 1
        else:
            if prediction[i]:
                fp += 1
            else:
                tn +=1

    precision = tp / (tp + fp)
    recall = tp / (fn + tp)
    f1 = 2 * recall * precision / (recall + precision)
    accuracy = (tp + tn) / (num_samples)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
