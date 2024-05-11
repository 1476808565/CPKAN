import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

# book最好效果，book原代码效果:testAuc: 0.7439 testF1: 0.6677     test auc: 0.7453 f1: 0.6663
# 50 1024  3  0.002 1e-4  32  64 64  test auc: 0.7529 f1: 0.6741
# 50 1024  3  0.002 1e-4  32  64 64  test auc: 0.7546 f1: 0.6772    emb_i:LR*2   triple:LR*2    User_0:Sigmoid*2
# 50 1024  3  0.002 1e-4  32  64 64  test auc: 0.7563 f1: 0.6763    emb_i:LR*2   triple:LR*2    User_0:Tanh、Sigmoid
# 50 1024  4  0.002 1e-4  128  64 32   test auc: 0.7609 f1: 0.6824    emb_i:ELU*2   triple:ELU*2    User_0:ReLU、Sigmoid
# 50 1024  4  0.002 1e-4  128  64 32   test auc: 0.7596 f1: 0.6818    emb_i:LR*2   triple:LR  User_0:ReLU、Sigmoid (调顺序)

# 上汽最好效果：test auc: 0.7725 f1: 0.7634。原代码效果：testAuc: 0.7657 testF1: 0.7502
# 50 2048  4  0.002 1e-5  32  64 32

# 下一步改LeakyReLU：
parser = argparse.ArgumentParser()
# shangqi
parser.add_argument('-d', '--dataset', type=str, default='book',
                    help='which dataset to use (music, book, movie, restaurant,shangqi)')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_layer', type=int, default=2, help='depth of layer')  # 4还没做
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')

parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=64, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=32, help='the number of triples in triple set of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

# book
# parser.add_argument('-d', '--dataset', type=str, default='book',
#                     help='which dataset to use (music, book, movie, restaurant,shangqi)')
# parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')  # 4还没做
# parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
#
# parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
# parser.add_argument('--user_triple_set_size', type=int, default=32, help='the number of triples in triple set of user')
# parser.add_argument('--item_triple_set_size', type=int, default=32, help='the number of triples in triple set of item')
# parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')
#
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)


if __name__ == '__main__':
    if not args.random_flag:
        set_random_seed(304, 2019)

    data_info = load_data(args)
    train(args, data_info)
