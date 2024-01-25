

def get_unique_labels(dataloader):
    labels = []
    for batch_idx, (x, y) in enumerate(dataloader):
        labels.extend(y.tolist())
    return list(set(labels))