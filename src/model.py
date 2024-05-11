import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(CPKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)

        ''''''  # 用（不注释）或者不用（注释掉）某些层都会影响效果,甚至改个声明的顺序都会影响结果（原顺序：emb_i、triple、User_0、u_i、item_i）
        self.triple_attention = nn.Sequential(  # LeakyReLU * 3
            nn.Linear(self.dim*2, self.dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.dim, 1, bias=True),
            nn.Sigmoid(),  # Sigmoid将样本值映射到0到1之间
        )
        self.emb_i_attention = nn.Sequential(  # Tanh * 2
            nn.Linear(self.dim*5, self.dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.dim, 1, bias=True),
            nn.Sigmoid(),  # Sigmoid将样本值映射到0到1之间
        )
        self.User_0_attention = nn.Sequential(  # ReLU * 2
            nn.Linear(self.dim, self.dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
            nn.Sigmoid(),
        )
        self.u_i_dig = nn.Sequential(  # LeakyReLU * 2
            nn.Linear(self.dim, self.dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
            nn.LeakyReLU(),
        )
        self.item_i_dig = nn.Sequential(  # LeakyReLU * 2
            nn.Linear(self.dim, self.dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
            nn.LeakyReLU(),
        )
        ''''''
        self._init_weight()  # 初始化的函数要放到所有网络层的最后用，不然self.XXX（某个网络层）会报错说object has no attribute 'XXX'
                
    def forward(
        self,
        items: torch.LongTensor,
        user_triple_set: list,  # [头/关系/尾][层][用户][交互物品]
        item_triple_set: list,
    ):       
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])  # 用户的交互物品
        # [batch_size, dim]
        # user_embeddings.append(user_emb_0.mean(dim=1))  # 取第1维度的平均值，如shape为（2,3,4），dim=1就是把2、4维的所有元素相加再取均值，然后返回（2,4）的矩阵

        ''''''
        new_user_emb_0 = self.User_0_attention(user_emb_0).mean(dim=1)
        user_embeddings.append(new_user_emb_0)
        ''''''

        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            att_user_emb_i = self.o_l_attention(user_emb_i, user_embeddings[0], user_embeddings[-1])

            ''''''
            # user_embeddings.append(user_emb_i)
            ''''''

            user_embeddings.append(self.u_i_dig(att_user_emb_i))
            att_user_dig_i = self.u_i_dig(att_user_emb_i)
            user_embeddings.append(att_user_dig_i)
            
        item_embeddings = []
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            att_item_emb_i = self.o_l_attention(item_emb_i, item_embeddings[0], item_embeddings[-1])

            ''''''
            # item_embeddings.append(item_emb_i)
            ''''''

            item_embeddings.append(self.item_i_dig(att_item_emb_i))
            att_item_dig_i = self.item_i_dig(att_item_emb_i)
            item_embeddings.append(att_item_dig_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
            # [batch_size, triple_set_size, dim]
            item_emb_0 = self.entity_emb(item_triple_set[0][0])
            # [batch_size, dim]
            item_embeddings.append(item_emb_0.mean(dim=1))
            
        scores = self.predict(user_embeddings, item_embeddings)
        return scores
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)
            
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
        
    def _init_weight(self):  # 用（不注释）或者不用（注释掉）某些层都会影响效果,甚至改个声明的顺序都会影响结果
        # init embedding
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        for layer in self.User_0_attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.triple_attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.emb_i_attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.u_i_dig:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.item_i_dig:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):

        hr = torch.mul(h_emb, r_emb)
        att_weights = self.triple_attention(torch.cat((hr, t_emb), dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights, dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)

        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i


    def o_l_attention(self, emb_i, origin_emb, last_layer_emb):

        oi = torch.mul(origin_emb, emb_i)
        li = torch.mul(last_layer_emb, emb_i)
        emb_i_att = self.emb_i_attention(torch.cat((emb_i,last_layer_emb,li, origin_emb,oi), dim=-1))
        att_emb_i = torch.mul(emb_i_att, emb_i)
        return att_emb_i
