from core.data_utils.dataset import CustomDGLDataset


def load_data(dataset, use_dgl=False, use_text=False, seed=0):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    # for finetuning LM
    data, text = get_raw_text(use_text=True, seed=seed)

    return data, num_classes, text
