import collections
import os
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
    return train_data, eval_data, test_data, n_entity, n_relation, user_triple_sets, item_triple_sets


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    # eval_ratio = 0
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    
    user_init_entity_set, item_init_entity_set = collaboration_propagation(rating_np, train_indices)
    
    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    
    return train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set
    
    
def collaboration_propagation(rating_np, train_indices):
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()  # user_history_item_dict[user]:[item1,item2,...]
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()  # 协作邻居字典
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []  # 存储用户历史交互的物品的列表
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []  # 存储与当前物品有过交互的用户的列表
            item_history_user_dict[item].append(user)
        
    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))  # 当前物品的协作邻居列表存入物品的字典

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:  # 如果物品没有协作邻居，那么把物品本身做为自己的协作邻居
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    print(n_entity)
    print(n_relation)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)  # kg[head]:[(tail1, relation1),(tail2, relation2),...]
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size] 
    triple_sets = collections.defaultdict(list)  # triple_sets[obj]:[(层数)[h, r, t],[h, r, t],...)]
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]  # 上一跳的尾实体变成新一跳的头实体

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                    
            if len(h) == 0:
                # del triple_sets[obj]
                # del kg[obj]
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))  # 把当前跳的所有头尾实体和关系作为元组保存
    return triple_sets
