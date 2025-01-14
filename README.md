# DGCDR

Pytorch implementation for paper _Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement_.

## Datasets
The dataset is processed based on [Amazon](https://amazon-reviews-2023.github.io/index.html)[[1]](#1), Douban[[2]](#2). 

Each domain contains a interaction file in the following format:

| Suffix | Content | Format |
|---|---|---|
| *.inter* | User-item interaction | user_id, item_id, rating |

## Training
You can use this command to train the model:

`python run_recbole_cdr.py`

The configuration settings are in a YAML file in  DGCDR/recbole_cdr/properties/.

## Acknowledgement
The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).


## References
<a id="1">[1]</a> Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024. Bridging Language and Items for Retrieval and Recommendation. arXiv preprint arXiv:2403.03952 (2024).

<a id="2">[2]</a> Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu
Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, et al., Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms,in: Proceedings of the 30th ACM International Conference on Information & Knowledge Management, Association for Computing Machinery, New York, NY, USA, 2021, pp. 4653–4664
