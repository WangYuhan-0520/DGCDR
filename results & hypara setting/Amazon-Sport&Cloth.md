# 1. Dataset Statistics
| Dataset      | #Users  | #items  | #Interactions | Sparsity |
|--------------|---------|---------|---------------|----------|
| Amazon-Sport | 149,520 | 94,057  | 1,474,955     | 99.9895% |
| Amazon-Cloth | 149,520 | 149,490 | 2,222,238     | 99.9901% |

Number of Overlapped User: 149,520

Number of Overlapped Item: 0


# 2.Hyper-parameters
### 2.1 Sport→Cloth
| Method            | Hyper-parameters                                                                                                                                                                                                               |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training**      | train_batch_size=2048<br/>epochs=400<br/>learning_rate=0.0001<br/>embedding_size=64<br/>mlp_hidden_size=[256]<br/>loss_type=BPR<br/>reg_weight=0.001<br/>drop_rate=0.1<br/>n_layers=3<br/>reg_weight=0.001<br/>init_way=xavier |
| **Model-related** | preference_disentangle=True<br/>cl_sim_weight=0.1<br/>cl_org_weight=0.1<br/>cl_decoder_weight=0.1<br/>item_cl_weight=0.1<br/>temperature=0.3<br/>connect_way=concat<br/>fuse_mode=attention                                    |

### 2.2 Cloth→Sport
| Method            | Hyper-parameters                                                                                                                                                                                                               |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training**      | train_batch_size=2048<br/>epochs=400<br/>learning_rate=0.0001<br/>embedding_size=64<br/>mlp_hidden_size=[256]<br/>loss_type=BPR<br/>reg_weight=0.001<br/>drop_rate=0.1<br/>n_layers=3<br/>reg_weight=0.001<br/>init_way=xavier |
| **Model-related** | preference_disentangle=True<br/>cl_sim_weight=0.01<br/>cl_org_weight=1<br/>cl_decoder_weight=0.1<br/>item_cl_weight=0.01<br/>temperature=0.25<br/>connect_way=concat<br/>fuse_mode=attention                                   |
