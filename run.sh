for dataset in 'cora'
do
    for seed in 0 1 2 3
    do
     WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
     python -m core.trainEnsemble dataset $dataset gnn.model.name MLP >> ${dataset}_mlp.out
     python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gcn.out
     python -m core.trainEnsemble dataset $dataset gnn.model.name GAT >> ${dataset}_gat.out
     python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage.out

done

