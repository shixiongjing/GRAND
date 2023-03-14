from __future__ import division
from __future__ import print_function

import time
import random
import argparse
import collections
import numpy as np
import sys
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, load_data_full, accuracy, sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN, MLP
from IGP import get_igp_train_set
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--dropfeature_rate', type=float, default=0.2,
                    help='Dropfeature rate (1 - keep probability).')
parser.add_argument('--dropedge_rate', type=float, default=0.2,
                    help='Dropedge rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1., help='constant controlling weight for consistency loss')
parser.add_argument('--beta', type=float, default=1., help='constant controlling weight for contrast loss')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample_n', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--sample_e', type=int, default=4, help='Sampling times of dropedge')
parser.add_argument('--sample_f', type=int, default=4, help='Sampling times of dropfeature')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--logname', default='df_', help='string for output log filename')
parser.add_argument('--shots',type = int, default = 20, help = 'N-way K-shots, max == default == 20. How many samples foe each classes.')
parser.add_argument('--active',action='store_true', default=False,
                    help='Disables CUDA training.')
#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def N_shot(idx,labels,shots):
    t_idx_dic = collections.defaultdict(list)
    label_list = labels.numpy()[:len(idx)]
    for i in range(len(label_list)):
        t_idx_dic[label_list[i]].append(idx.numpy()[i])
    
    new_idx_train = []
    for k in t_idx_dic.keys():
        new_idx_train += random.sample(t_idx_dic[k],k=shots)

    new_idx_train = torch.LongTensor(new_idx_train)
    return new_idx_train





# Load data
A, adj, or_adj, features, labels, idx_train, idx_val, idx_test, edges = load_data_full(dataset)

real_labels = copy.deepcopy(labels)
real_labels = real_labels.cuda()
#if args.shots == args.shots: # check for NaN value
new_idx_train = N_shot(idx_train,labels,args.shots)
#if args.active:
new_idx_train, new_labels = get_igp_train_set(or_adj, features, labels, new_idx_train, idx_val, idx_test, args)
idx_train = new_idx_train

idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)

# Model and optimizer
model = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn = args.use_bn)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()


def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()

def propagate_e(feature, a, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    #x = feature
    y = torch.spmm(a,feature).detach_()
    x = y
    for i in range(order):
        x = torch.spmm(a, x).detach_()
        y.add_(x)
    return y.div_(order+1.0).detach_()

def preprocess(a):
    #d1 = np.array(a.sum(axis-1))**(-0.5)
    #d2 = np.array(a.sum(axis=0))**(-0.5)
    D1_ = np.array(a.sum(axis=1))**(-0.5)
    D2_ = np.array(a.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = a.dot(D1_)
    A_ = D2_.dot(A_)
    A_ = sparse_mx_to_torch_sparse_tensor(A_) 
    if args.cuda:
        A_ = A_.cuda()
    return A_    

def random_edge_sample(edges, droprate):
    edges = list(edges)
    n = features.shape[0]
    m = len(edges)
    index = np.random.permutation(m)
    percent = 1. - droprate
    preserve_num = int(m * percent)
     
    index_ = index[:preserve_num]
    sample_row = [edges[x][0] for x in index_]
    sample_col = [edges[x][1] for x in index_]
    sample_adj = sp.csr_matrix((np.ones(preserve_num), (sample_row, sample_col)), shape=(n,n))
    sample_adj = sample_adj + sample_adj.T.multiply(sample_adj.T>sample_adj) - sample_adj.multiply(sample_adj.T>sample_adj) + sp.eye(n)
    sample_adj = preprocess(sample_adj)
    return sample_adj


def rand_node_prop(features, training):
    n = features.shape[0]
    drop_node_rate = args.dropnode_rate

    if training:

        drop_node_rates = torch.FloatTensor(np.ones(n) * drop_node_rate)
            
        masks = torch.bernoulli(1. - drop_node_rates).unsqueeze(1)

        features = masks.cuda() * features

            
    else:
            
        features = features * (1. - drop_node_rate)
    
    features = propagate(features, A, args.order)    
    return features

def rand_feature_prop(features, training):
    t = features.shape[1]
    drop_feature_rate = args.dropfeature_rate

    if training:
        drop_feature_rates = torch.FloatTensor(np.ones(t) * drop_feature_rate)

        masks2 = torch.bernoulli(1. - drop_feature_rates)
        features = masks2.cuda() * features
    else:
        features = features * (1. - drop_feature_rate)

    features = propagate(features, A, args.order)    
    return features

def rand_edge_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropedge_rate
    #drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    if training: 
        a = random_edge_sample(edges, drop_rate)
        #a = sparse_dropout(A, training, drop_rate)
    else:
        a = A#preprocess(adj)
    features = propagate_e(features, a, args.order)    
    return features


def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def train(epoch):
    t = time.time()
    
    X = features
    
    model.train()
    optimizer.zero_grad()
    loss_train = 0.

    # Process original data inference
    org_X = rand_node_prop(X, training=False)
    org_output = torch.log_softmax(model(org_X), dim=-1)
    pos_pair = 0.
    neg_pair = 0.

    K_n = args.sample_n
    K_e = args.sample_e
    K_f = args.sample_f
    if not K_n + K_e + K_f > 0:
        print('Sampling for edge, node, feature drop is all zero, exit program...')
        quit()

    # Add node drops
    loss_consis_node = 0
    if K_n > 0 and args.dropnode_rate > 0:
        X_list = []
        for k in range(K_n):
            X_list.append(rand_node_prop(X, training=True))

        output_list = []
        for k in range(K_n):
            output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

        for k in range(K_n):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])

        for x in idx_train:
            for y in idx_train:
                for k in range(K_n):
                    if labels[x]==labels[y]:
                        pos_pair += F.cross_entropy(output_list[k][x], org_output[y])
                    else:
                        neg_pair += F.cross_entropy(output_list[k][x], org_output[y])

        loss_consis_node = consis_loss(output_list)





    # Add edge drops
    loss_consis_edge = 0
    if K_e > 0 and args.dropedge_rate > 0:
        X_list = []
        
        for k in range(K_e):
            X_list.append(rand_edge_prop(X, training=True))

        output_list = []
        for k in range(K_e):
            output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

        for k in range(K_e):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])

        for x in idx_train:
            for y in idx_train:
                for k in range(K_e):
                    if labels[x]==labels[y]:
                        pos_pair += F.cross_entropy(output_list[k][x], org_output[y])
                    else:
                        neg_pair += F.cross_entropy(output_list[k][x], org_output[y])

        loss_consis_edge = consis_loss(output_list)

    # Add feature drops
    loss_consis_feature = 0
    if K_f > 0 and args.dropfeature_rate > 0:
        X_list = []
        
        for k in range(K_f):
            X_list.append(rand_feature_prop(X, training=True))

        output_list = []
        for k in range(K_f):
            output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

        for k in range(K_f):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])

        for x in idx_train:
            for y in idx_train:
                for k in range(K_n):
                    if labels[x]==labels[y]:
                        pos_pair += F.cross_entropy(output_list[k][x], org_output[y])
                    else:
                        neg_pair += F.cross_entropy(output_list[k][x], org_output[y])

        loss_consis_feature = consis_loss(output_list)
        
    
    
    




        
    loss_train = loss_train/(K_n + K_e + K_f)
    loss_consis = (loss_consis_node * K_n + loss_consis_edge * K_e + loss_consis_feature * K_f) / (K_n + K_e + K_f)
    loss_contrast = pos_pair/neg_pair
    #loss_train = F.nll_loss(output_1[idx_train], labels[idx_train]) + F.nll_loss(output_1[idx_train], labels[idx_train])
    #loss_js = js_loss(output_1[idx_unlabel], output_2[idx_unlabel])
    #loss_en = entropy_loss(output_1[idx_unlabel]) + entropy_loss(output_2[idx_unlabel])
    

    loss_train = loss_train + args.alpha * loss_consis + args.beta * loss_contrast
    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluation Stage
    model.eval()

    
    X = rand_node_prop(X,training=False)
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_val = F.nll_loss(output[idx_val], real_labels[idx_val]) 
    acc_val = accuracy(output[idx_val], real_labels[idx_val])
    

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()


def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset +'.pkl'))



def test():
    model.eval()
    X = features
    X = rand_node_prop(X, training=False)
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], real_labels[idx_test])
    acc_test = accuracy(output[idx_test], real_labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
    log_result = ("dataset: {}, "
              "lr: {}, "   
              "epochs: {}, "   
              "hidden: {}, "    
              "tem: {}, "  
              "lam: {}, " 
              "dropnode_rate: {}, "
              "dropfeature_rate: {}, "
              "dropedge_rate: {}, "
              "Loss: {:.4f}, "
              "TestAcc: {:.4f}, "
              "Overall\n").format(args.dataset,
                                  args.lr,
                                  args.epochs,
                                  args.hidden,
                                  args.tem,
                                  args.lam, 
                                  args.dropnode_rate,
                                  args.dropfeature_rate,
                                  args.dropedge_rate,
                                  loss_test.item(),
                                  acc_test.item())
    log_result_pre = ( 
              "{}, "
              "{}, "
              "{:.4f}, "
              "\n").format(
                                  args.dropnode_rate,
                                  args.dropfeature_rate,
                                  acc_test.item())
    if args.shots <= 20: # check if few shot is used
        with open('{}/result/{}_{}_result_log_{}-shot.txt'.format(sys.path[0], args.logname, args.dataset, args.shots), 'a') as f:
            f.write(log_result)
    else:
        with open('{}/result/{}_{}_result_log.txt'.format(sys.path[0], args.logname, args.dataset), 'a') as f:
            f.write(log_result)



Train()
test()
