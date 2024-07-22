import os
from util.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
from networks.word2sequence import Word2Sequence
import torch
# cwd=os.getcwd()
cwd, _ = os.path.split(os.getcwd())
cwd, _ = os.path.split(cwd)

################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = './data/'+dataname+'/data.TD_RvNN.vol_5000.txt'
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    # 'D:\\paper_reproduction\\05_Rumor Detection on Social Media with Directional Graph\\BIGCN\\data\\Twitter15graph'
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)# 过滤掉了只有一个节点的树
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path) # 过滤掉了只有一个节点的树
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData2(dataname, treeDic, fold_x_train, fold_x_test, droprate):

    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, dataname=dataname)# 过滤掉了只有一个节点的树
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic, dataname=dataname) # 过滤掉了只有一个节点的树
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def sup_loss_cal(features, labels=None,mask=None,temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):

    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0] # 128
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 形状为[128,128]，相同的标签为1，不同的标签为0,包括真身的也为1
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # 多少个增强视图
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # unbind去掉一个维度，最后形状[128,256]
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature # [128,256]
        anchor_count = contrast_count # 1
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits # div除法， matmul举证相乘  2n个样本，依次分别与2n个样本计算相似度 [256,256]
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 按行得到每行最大值[256,1]
    logits = anchor_dot_contrast - logits_max.detach()  # 每一行分别减去 每行的最大值,是为了数据平稳？

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # 行和列分别复制一次 [batch_size，batch_size]
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )  # 自身位置置为0，其余为1 比如第一行的第一个元素为0 第二行的第二个元素为0
    mask = mask * logits_mask  # 对应位置相乘，点乘 [128,128]，自身的位置置为0

    # compute log_prob 计算相同的标签 [256,256]
    exp_logits = torch.exp(logits) * logits_mask  # 除了自身的值
    # 把对比学习的损失化简了，损失函数的分母为所有节点对的的相似度，分子为正样本对的相似度，这里所有的样本减去一个相似度，相当于处于
    # 对数（a/b）=对数（a）-对数（b）
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 每一行的值分别减去 每一行的和
    # 把正样本对挑选出来
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos

    # print(torch.isnan(loss).any())
    # print(torch.isinf(loss).any())
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def pro_loss_cal(features, index,cluster,temperature=0.07):

    # 获取样本所属原型id
    prototypes = cluster['centroids']
    density = cluster['density']
    im2cluster = cluster['im2cluster']
    proto_id = im2cluster[index]

    sim_matrix = torch.matmul(features, prototypes.T) # [128,12]
    sim_matrix = torch.exp(sim_matrix / density)

    pos_sim = sim_matrix[range(features.shape[0]), proto_id]

    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


