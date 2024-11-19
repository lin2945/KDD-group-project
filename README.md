This is the code for the KDD group project.

For the Cora dataset, please refer to https://github.com/XiaoxinHe/TAPE. The python environment setup could also be find there.

Use run.sh to run the codes.

To finetune the LM, 
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset cora
```

To train certain GNN model,(MLP as example)
```
python -m core.trainGNN gnn.model.name MLP
```

