
r"""
DCCDR
################################################
Reference:
    Zhang et al. "Disentangled Contrastive Learning for Cross-Domain Recommendation." in DASFAA 2023.
"""

import numpy as np
import scipy.sparse as sp

import torch

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
from recbole.model.loss import BPRLoss


class DCCDR(CrossDomainRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DCCDR, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']
        self.temp = config['temp']
        self.embed_size = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        self.A_split = config['A_split']  # str type: connect way for all layers
        self.ssl_weight = config['ssl_weight']
        self.folds = 10  # 将邻接矩阵A分割成多个子矩阵并转换为稀疏张量。处理大规模图时为True可以节省内存并提高计算效率，需要设置folds数（每一段子矩阵的行数）

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users,
                                                        embedding_dim=self.embed_size, device=self.device)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users,
                                                        embedding_dim=self.embed_size, device=self.device)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items,
                                                        embedding_dim=self.embed_size, device=self.device)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items,
                                                        embedding_dim=self.embed_size, device=self.device)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.dropout = torch.nn.Dropout(p=self.drop_rate)

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(
            np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(
            np.float32)
        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,
                                                            self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,
                                                            self.total_num_items).to(self.device)

        self.source_user_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=1)).to(self.device)
        self.target_user_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=1)).to(self.device)
        self.source_item_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=0)).transpose(0, 1).to(
            self.device)
        self.target_item_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=0)).transpose(0, 1).to(
            self.device)

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def disentangle(self, all_embeddings, norm_adj_matrix, n_factors_l=2):
        all_embs_tp = torch.split(all_embeddings, int(self.embed_size / n_factors_l), 1)
        disentangle_embs = []  # 原项目中all_embs
        for i in range(n_factors_l):
            disentangle_emb = torch.sparse.mm(norm_adj_matrix, all_embs_tp[i])
            disentangle_embs.append(self.dropout(disentangle_emb))
        new_all_embeddings = torch.cat([disentangle_embs[0], disentangle_embs[1]], dim=1)
        return disentangle_embs, new_all_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        n_factors_l = 2  # 将特征解耦为n个部分
        source_all_disentangle_embs = []
        target_all_disentangle_embs = []
        for layer_idx in range(self.n_layers):
            # 在每层卷积中进行common和specific特征的操作并concat
            source_disentangle_embs, source_layer_embeddings = self.disentangle(source_all_embeddings,
                                                                                source_norm_adj_matrix, n_factors_l)
            target_disentangle_embs, target_layer_embeddings = self.disentangle(target_all_embeddings,
                                                                                target_norm_adj_matrix, n_factors_l)
            source_all_disentangle_embs.append(source_disentangle_embs)
            target_all_disentangle_embs.append(target_disentangle_embs)
            source_embeddings_list.append(source_layer_embeddings)
            target_embeddings_list.append(target_layer_embeddings)
        # mean vs. concat (原论文为concat，而代码中为mean，保留代码版本)
        source_lightgcn_all_embeddings = torch.mean(torch.stack(source_embeddings_list, dim=1), dim=1)
        target_lightgcn_all_embeddings = torch.mean(torch.stack(source_embeddings_list, dim=1), dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                             [self.total_num_users,
                                                                              self.total_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                             [self.total_num_users,
                                                                              self.total_num_items])

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings, \
               source_all_disentangle_embs[-1], target_all_disentangle_embs[-1]

    def ssl_loss_strategy(self, source_all_disentangle_embs, target_all_disentangle_embs):
        invariant_embed_1, specific_embed_1 = source_all_disentangle_embs[0], source_all_disentangle_embs[1]
        invariant_u_embed_1, specific_u_embed_1 = invariant_embed_1[:self.total_num_users], specific_embed_1[
                                                                                            :self.total_num_users]
        invariant_embed_2, specific_embed_2 = target_all_disentangle_embs[0], target_all_disentangle_embs[1]
        invariant_u_embed_2, specific_u_embed_2 = invariant_embed_2[:self.total_num_users], specific_embed_2[
                                                                                            :self.total_num_users]

        # 正则化
        normalize_invariant_user_1 = torch.nn.functional.normalize(invariant_u_embed_1, p=2, dim=1)
        normalize_invariant_user_2 = torch.nn.functional.normalize(invariant_u_embed_2, p=2, dim=1)
        normalize_specific_user_1 = torch.nn.functional.normalize(specific_u_embed_1, p=2, dim=1)
        normalize_specific_user_2 = torch.nn.functional.normalize(specific_u_embed_2, p=2, dim=1)

        # 对比函数计算
        pos_score_user = torch.sum(torch.mul(normalize_invariant_user_1, normalize_invariant_user_2), dim=1)
        neg_score_1 = torch.sum(torch.mul(normalize_invariant_user_1, normalize_specific_user_1), dim=1)
        neg_score_2 = torch.sum(torch.mul(normalize_invariant_user_2, normalize_specific_user_2), dim=1)
        neg_score_3 = torch.sum(torch.mul(normalize_specific_user_1, normalize_specific_user_2), dim=1)
        # neg_score_4 = torch.matmul(normalize_invariant_user_1, normalize_invariant_user_2.T)  # for what？

        pos_score = torch.exp(pos_score_user / self.temp)
        neg_score_1 = torch.exp(neg_score_1 / self.temp)
        neg_score_2 = torch.exp(neg_score_2 / self.temp)
        neg_score_3 = torch.exp(neg_score_3 / self.temp)
        # neg_score_4 = torch.sum(torch.exp(neg_score_4 / self.temp), dim=1)
        
        # 此前项目里的是torch.sum，将所有user的loss相加，应该是所有用户的平均loss，修改为torch.mean
        ssl_loss_user = -torch.mean(torch.log(pos_score / (neg_score_1 + neg_score_2 + neg_score_3 + pos_score)))
        return ssl_loss_user

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings, \
        source_all_disentangle_embs, target_all_disentangle_embs = self.forward()

        losses = []
        # 计算源领域BPR_loss
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

        source_user_e = source_user_all_embeddings[source_user]
        source_item_e = source_item_all_embeddings[source_item]
        source_neg_item_e = source_item_all_embeddings[source_neg_item]

        source_pos_output = torch.mul(source_user_e, source_item_e).sum(dim=1)
        source_neg_output = torch.mul(source_user_e, source_neg_item_e).sum(dim=1)
        source_loss = self.rec_loss(source_pos_output, source_neg_output) + \
                      self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                      self.source_item_embedding(source_item),
                                                      self.source_item_embedding(source_neg_item))
        losses.append(source_loss)
        # 计算目标领域BPR_loss
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

        target_user_e = target_user_all_embeddings[target_user]
        target_item_e = target_item_all_embeddings[target_item]
        target_neg_item_e = target_item_all_embeddings[target_neg_item]
        target_pos_output = torch.mul(target_user_e, target_item_e).sum(dim=1)
        target_neg_output = torch.mul(target_user_e, target_neg_item_e).sum(dim=1)
        target_loss = self.rec_loss(target_pos_output, target_neg_output) + \
                      self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                      self.target_item_embedding(target_item),
                                                      self.target_item_embedding(target_neg_item))
        losses.append(target_loss)
        # 计算对比损失
        ssl_loss = self.ssl_loss_strategy(source_all_disentangle_embs, target_all_disentangle_embs)
        losses.append(self.ssl_weight * ssl_loss)

        return tuple(losses)

    def predict(self, interaction):
        result = []
        _, _, target_user_all_embeddings, target_item_all_embeddings, _, _, = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e, _, _, = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e
