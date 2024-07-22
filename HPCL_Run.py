import sys, os

sys.path.append(os.getcwd())
from util.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from util.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from util.loadSplitData import *
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import faiss
import torch.nn as nn
import math
from torch.utils.data import Subset
import warnings
import random
from networks.HPCLEncoder import Net
import argparse

th.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def compute_features(computer_feature_loader, model, dim):
    print('Computing features...')
    model.eval()
    features = th.zeros(len(computer_feature_loader.dataset), dim).cuda()
    label = []
    for i, (index, Batch_data) in enumerate(tqdm(computer_feature_loader)):
        with th.no_grad():
            Batch_data.to(device)
            _, feat = model(Batch_data)  # resnet50 对图片进行编码，batch_size 40，最终形状为[40,128]
            features[index] = feat
            label.extend(Batch_data.y.cpu().detach().numpy().tolist())
    return features.cpu(), label


def run_kmeans(x, temperature, num_cluster, label):
    """
    Args:
        x: data to be clustered
    """
    print('performing kmeans clustering')
    # results = {'im2cluster': [], 'centroids': [], 'density': []}

    """
    分为4个类别，也是是4个群

    分别对每个类别的集群进行k-means,记录每个类别的索引

    """
    label_0_index = []
    label_1_index = []
    label_2_index = []
    label_3_index = []
    for index, item in enumerate(label):
        """
        每个类别的索引
        """
        if item == 0:
            label_0_index.append(index)
        if item == 1:
            label_1_index.append(index)
        if item == 2:
            label_2_index.append(index)
        if item == 3:
            label_3_index.append(index)
    x_label_0 = x[label_0_index]  # 获取特征
    x_label_1 = x[label_1_index]
    x_label_2 = x[label_2_index]
    x_label_3 = x[label_3_index]

    result_0 = get_k_means_result(x_label_0, num_cluster, temperature)
    result_1 = get_k_means_result(x_label_1, num_cluster, temperature)
    result_2 = get_k_means_result(x_label_2, num_cluster, temperature)
    result_3 = get_k_means_result(x_label_3, num_cluster, temperature)

    centroids = th.cat((result_0['centroids'], result_1['centroids'], result_2['centroids'], result_3['centroids']), 0)
    density = th.cat((result_0['density'], result_1['density'], result_2['density'], result_3['density']), 0)
    im2cluster_0 = result_0['im2cluster'] + 0
    im2cluster_1 = result_1['im2cluster'] + num_cluster
    im2cluster_2 = result_2['im2cluster'] + num_cluster * 2
    im2cluster_3 = result_3['im2cluster'] + num_cluster * 3

    im2cluster = th.tensor([-1 for i in range(len(label))])
    for index_enu, index_label in enumerate(label_0_index):
        im2cluster[index_label] = im2cluster_0[index_enu]

    for index_enu, index_label in enumerate(label_1_index):
        im2cluster[index_label] = im2cluster_1[index_enu]

    for index_enu, index_label in enumerate(label_2_index):
        im2cluster[index_label] = im2cluster_2[index_enu]

    for index_enu, index_label in enumerate(label_3_index):
        im2cluster[index_label] = im2cluster_3[index_enu]

    results = {'centroids': centroids, 'density': density, 'im2cluster': im2cluster}

    return results


def get_k_means_result(x, num_cluster, temperature):
    results = {}
    # intialize faiss clustering parameters
    d = x.shape[1]  # 特征维度 256
    k = int(num_cluster)  # 集群个数 3
    clus = faiss.Clustering(d, k)
    clus.verbose = True  # 使聚类更详细
    clus.niter = 20  # 聚类迭代次数
    clus.nredo = 5  # 重复聚类的次数
    clus.seed = 0
    clus.max_points_per_centroid = 1000  # 最大样本个数
    clus.min_points_per_centroid = 10  # 最小样本个数

    res = faiss.StandardGpuResources()  # 使用单GPU运行
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)  # 训练
    # D表示每个样本到原型的距离，I表示每个样本所属的簇
    D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]  # 每个样本的所属簇

    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)  # 聚类原型向量

    # sample-to-centroid distances for each cluster
    Dcluster = [[] for c in range(k)]  # 保存每个样本到原型的距离
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    # concentration estimation (phi)
    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)  # 计算每个类的密度，d越小，密度越大
            density[i] = d
            # 如果聚类只有一个节点，使用最大值来表示密度
    # if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax
            # 设定密度的区间，在最大值和最小值之间，如果小于10百分位，则取10百分位的值，如果大于90百分位，则取90百分位的值
    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    density = temperature * density / density.mean()  # 相当于密度乘以（温度/均值）  #scale the mean to temperature

    # convert to cuda Tensors for broadcast
    centroids = th.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)  # 二范数归一化

    im2cluster = th.LongTensor(im2cluster).cuda()
    density = th.Tensor(density).cuda()

    results['centroids'] = centroids  # 原型向量 [4,256]
    results['density'] = density  # [4,]
    results['im2cluster'] = im2cluster  # [299,]

    return results


def run_model(treeDic, x_test, x_train, droprate, lr, weight_decay, patience, n_epochs, batchsize,
              dataname, word_embedding_dim, post_embedding_dim, word_head, post_head, word_dff, post_dff,
              word_droupout,
              post_droupout, word_N, post_N, num_words, num_posts, out_dim, in_feats, hid_feats, out_feats):
    model = Net(dataname=dataname, batch_size=batchsize, in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats,
                word_embedding_dim=word_embedding_dim, post_embedding_dim=post_embedding_dim, word_head=word_head,
                post_head=post_head, word_dff=word_dff, post_dff=post_dff, word_droupout=word_droupout,
                post_droupout=post_droupout, word_N=word_N, post_N=post_N, num_words=num_words, num_posts=num_posts,
                out_dim=out_dim).to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 如果不用early_stopping 效果如何？
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadBiData2(dataname, treeDic, x_train, x_test, droprate)
    computer_feature_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers=1)
    for epoch in range(n_epochs):

        features, label = compute_features(computer_feature_loader, model, 256)
        features = F.normalize(features, dim=1)  # l2 规范化

        reps = features.numpy()  # [1176, 128]
        cluster_result = run_kmeans(reps, temperature=0.2, num_cluster=4, label=label)

        """
        计算样本与原型的距离
        计算每个样本的得分
        然后排序,难度是从容易到难，值越小，越容易
        """
        distances = th.cdist(features.to(device), cluster_result['centroids'], p=2)
        pos_distance = distances[range(features.shape[0]), cluster_result['im2cluster']]
        score = pos_distance / (distances.sum(dim=1) - pos_distance)
        sorted_indices = th.argsort(score)

        """
        确定采样的数量
        求模
        """
        TS_n = 0.5
        TS_T = 4
        TS = 1 if (epoch + 1) % TS_T == 0 else TS_n + ((1 - TS_n) / TS_T * ((epoch + 1) % TS_T))

        num_selection = math.ceil(TS * features.shape[0])

        """
        实施依据权重，随机采样
        """
        random_elements = set()

        while len(random_elements) < num_selection:
            element = random.choices(range(features.shape[0]), weights=score, k=num_selection)
            for temp in element:
                random_elements.add(temp)

        if len(random_elements) > num_selection:
            random_elements = random.sample(list(random_elements), num_selection)
        random_elements = list(random_elements)  # 转换为列表形式（可选）

        """
        设置 train_loader
        """
        new_dataset = Subset(traindata_list, random_elements)

        train_loader = DataLoader(new_dataset, batch_size=batchsize, shuffle=True, num_workers=1)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)

        for (element_index, Batch_data) in tqdm_train_loader:
            Batch_data.to(device)
            out_labels, x_feature = model(Batch_data)

            x_feature = F.normalize(x_feature, dim=1)
            # features_cl = x_feature.unsqueeze(1)

            """
            filter tensor,去掉只有一个实例实例的类别特征，因为在有监督对比学习中，该实例无法组成正样本对
            """
            sup_loss_y = Batch_data.y.cpu().detach().numpy().tolist()

            # 计算每个元素在整个列表中的出现次数
            element_counts = {item: sup_loss_y.count(item) for item in sup_loss_y}

            # 找到元素数量为1的值
            values_with_single_count = [item for item, count in element_counts.items() if count == 1]

            # 找到元素数量为1的值对应的索引
            indices_with_single_count = [index for index, item in enumerate(sup_loss_y) if
                                         item in values_with_single_count]

            # 从列表中去除元素数量为1的值
            filtered_list = [item for index, item in enumerate(sup_loss_y) if index not in indices_with_single_count]

            sup_loss_y = th.tensor(filtered_list).to(device)

            filtered_tensor = x_feature[
                [idx for idx in range(x_feature.size(0)) if idx not in indices_with_single_count]]

            features_cl = filtered_tensor.unsqueeze(1)
            # end

            loss_cl = sup_loss_cal(features=features_cl, labels=sup_loss_y)
            loss_pro = pro_loss_cal(features=x_feature, index=element_index, cluster=cluster_result)

            finalloss = F.nll_loss(out_labels, Batch_data.y)

            # loss = finalloss
            loss = finalloss + 0.1 * (loss_cl + loss_pro)
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []

        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        flag = 0
        for (element_index, Batch_data) in tqdm_test_loader:
            Batch_data.to(device)

            val_out, x_feature = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)

            temp_val_accs.append(val_acc)

            flag = flag + 1

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))

        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), model, 'HPCL', dataname)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(val_accs)

    return train_losses, val_losses, train_accs, val_accs


def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("seed:", seed)

parser = argparse.ArgumentParser(description='HPCL')
parser.add_argument('--lr', default=0.0005, type=float, help='Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay coefficient')
parser.add_argument('--patience', default=10, type=int, help='Early Stopping')
parser.add_argument('--n_epochs', default=200, type=int, help='Training Epochs')
parser.add_argument('--batchsize', default=128, type=int, help='Batch Size')
parser.add_argument('--droprate', default=0.2, type=float, help='Randomly invalidate some edges')
parser.add_argument('--seed', default=1, type=int)

parser.add_argument('--in_feats', default=5000, type=int)
parser.add_argument('--hid_feats', default=64, type=int)
parser.add_argument('--out_feats', default=64, type=int)

parser.add_argument('--word_embedding_dim', default=128, type=int)
parser.add_argument('--post_embedding_dim', default=128, type=int)
parser.add_argument('--word_head', default=2, type=int)
parser.add_argument('--post_head', default=2, type=int)
parser.add_argument('--word_dff', default=128, type=int)
parser.add_argument('--post_dff', default=128, type=int)
parser.add_argument('--word_droupout', default=0.2, type=float)
parser.add_argument('--post_droupout', default=0.2, type=float)
parser.add_argument('--word_N', default=1, type=int)
parser.add_argument('--post_N', default=1, type=int)
parser.add_argument('--num_words', default=35, type=int)
parser.add_argument('--num_posts', default=20, type=int)
parser.add_argument('--out_dim', default=128, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    set_seed(args.seed)
    datasetname = "Twitter16" # Twitter15 Twitter16
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_set, train_set = loadSplitData(datasetname)
    treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs = run_model(treeDic=treeDic, x_test=test_set, x_train=train_set,
                                                               droprate=args.droprate, lr=args.lr,
                                                               weight_decay=args.weight_decay, patience=args.patience,
                                                               n_epochs=args.n_epochs,
                                                               batchsize=args.batchsize, dataname=datasetname,
                                                               word_embedding_dim=args.word_embedding_dim,
                                                               post_embedding_dim=args.post_embedding_dim,
                                                               word_head=args.word_head,
                                                               post_head=args.post_head, word_dff=args.word_dff,
                                                               post_dff=args.post_dff,
                                                               word_droupout=args.word_droupout,
                                                               post_droupout=args.post_droupout, word_N=args.word_N,
                                                               post_N=args.post_N, num_words=args.num_words,
                                                               num_posts=args.num_posts,
                                                               out_dim=args.out_dim, in_feats=args.in_feats,
                                                               hid_feats=args.hid_feats,
                                                               out_feats=args.out_feats)

    print("Total_Best_Accuracy:{}".format(max(val_accs)))



