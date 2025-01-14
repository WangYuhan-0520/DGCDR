# 1. Dataset Statistics
| Dataset      | #Users | #items | #Interactions | Sparsity |
|--------------|--------|--------|---------------|----------|
| Amazon-Elec  | 35,827 | 62,548 | 811,969       | 99.9638% |
| Amazon-Cloth | 35,827 | 72,669 | 847,042       | 99.9675% |

Number of Overlapped User: 35,827

Number of Overlapped Item: 0


# 2.Hyper-parameters
### 2.1 Elec→Cloth
| Method            | Hyper-parameters                                                                                                                                                                                                               |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training**      | train_batch_size=4096<br/>epochs=400<br/>learning_rate=0.0001<br/>embedding_size=64<br/>mlp_hidden_size=[256]<br/>loss_type=BPR<br/>reg_weight=0.001<br/>drop_rate=0.1<br/>n_layers=3<br/>reg_weight=0.001<br/>init_way=xavier |
| **Model-related** | preference_disentangle=True<br/>cl_sim_weight=0.1<br/>cl_org_weight=0.1<br/>cl_decoder_weight=0.1<br/>item_cl_weight=0.1<br/>temperature=0.15<br/>connect_way=concat<br/>fuse_mode=attention                                   |

### 2.2 Cloth→Elec
| Method            | Hyper-parameters                                                                                                                                                                                                               |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training**      | train_batch_size=4096<br/>epochs=400<br/>learning_rate=0.0001<br/>embedding_size=64<br/>mlp_hidden_size=[256]<br/>loss_type=BPR<br/>reg_weight=0.001<br/>drop_rate=0.1<br/>n_layers=3<br/>reg_weight=0.001<br/>init_way=xavier |
| **Model-related** | preference_disentangle=True<br/>cl_sim_weight=0.01<br/>cl_org_weight=1<br/>cl_decoder_weight=0.1<br/>item_cl_weight=0.01<br/>temperature=0.15<br/>connect_way=concat<br/>fuse_mode=attention                                   |
