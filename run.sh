for dataset in 'cora'
do
    for seed in 0 1 2 3
    do
     WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
#   # python -m core.trainEnsemble dataset $dataset gnn.model.name MLP >> ${dataset}_mlp.out
#      python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gcn.out
      #python -m core.trainEnsemble dataset pubmed gnn.model.name MLP >> pubmed_mlp.out
#     python -m core.trainEnsemble dataset cora gnn.model.name GCN >> cora_gcn.out
#      python -m core.trainGNN dataset cora gnn.model.name GCN >> cora_gcn.out
#
#   # python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage.out
#   # python -m core.trainEnsemble dataset $dataset gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${dataset}_revgat.out
done

