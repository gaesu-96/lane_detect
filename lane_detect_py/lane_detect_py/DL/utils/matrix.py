import torch
import torchmetrics

def binary_evaluate_metrics(preds, targets, device, acc=False, pre=False, re=False, f1=False, di=False, io=False):
    # Move tensors to the same device
    preds = preds.to(device)
    targets = targets.to(device)

    # Convert probabilities to binary predictions
    preds = preds > 0.5
    preds = preds.int()
    targets = targets.int()

    results = {}

    if acc:
        accuracy = torchmetrics.classification.Accuracy(task='binary').to(device)
        accuracy.update(preds, targets)
        results['Accuracy'] = accuracy.compute()

    if pre:
        precision = torchmetrics.classification.Precision(num_classes=1, average='binary', task='binary').to(device)
        precision.update(preds, targets)
        results['Precision'] = precision.compute()

    if re:
        recall = torchmetrics.classification.Recall(num_classes=1, average='binary', task='binary').to(device)
        recall.update(preds, targets)
        results['Recall'] = recall.compute()

    if di:
        dice = torchmetrics.classification.Dice(num_classes=1, average='binary', multiclass=False, task='binary').to(device)
        dice.update(preds, targets)
        results['Dice'] = dice.compute()

    if io:
        iou = torchmetrics.classification.JaccardIndex(num_classes=1, average='binary', task='binary').to(device)
        iou.update(preds, targets)
        results['IoU'] = iou.compute()

    return results


def multiclass_evaluate_metrics(preds, targets, device, acc=False, pre=False, re=False, f1=False, di=False, io=False):
    # Move tensors to the same device
    preds = preds.to(device)
    targets = targets.to(device)

    # Convert probabilities to binary predictions
    results = {}

    if acc:
        accuracy = torchmetrics.Accuracy(num_classes=5, average='macro', task='multiclass').to(device)
        accuracy.update(preds, targets)
        results['Accuracy'] = accuracy.compute().item()

    if pre:
        precision = torchmetrics.Precision(num_classes=5, average='macro', task='multiclass').to(device)
        precision.update(preds, targets)
        results['Precision'] = precision.compute().item()

    if re:
        recall = torchmetrics.Recall(num_classes=5, average='macro', task='multiclass').to(device)
        recall.update(preds, targets)
        results['Recall'] = recall.compute().item()

    if f1:
        f1_score = torchmetrics.F1Score(num_classes=5, average='macro', task='multiclass').to(device)
        f1_score.update(preds, targets)
        results['F1 Score'] = f1_score.compute().item()

    if io:
        iou = torchmetrics.JaccardIndex(num_classes=5, average='macro', task='multiclass').to(device)
        iou.update(preds, targets)
        results['IoU'] = iou.compute().item()

    return results
