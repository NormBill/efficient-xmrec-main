from xmrec.utils.forec_utils import *
from xmrec.data.data import CentralIDBank
import pickle
import numpy as np
import torch

def prototype_embedding(user_indices_tensor):
    user_indices_list = user_indices_tensor.tolist()
    nearest_centers = []

    # 1. 读取所有的cluster centers
    with open("/content/efficient-xmrec-main/DATA2/proc_data/cluster_centers.txt", "r") as f:
        cluster_centers = [list(map(float, line.strip().split())) for line in f.readlines()]
    cluster_centers = np.array(cluster_centers)

    # 从txt文件加载映射关系
    index_to_user_id_mapping = {}
    with open("/content/index_to_user_id.txt", "r") as f:
        for line in f:
            index, userid = line.strip().split("\t")
            index_to_user_id_mapping[int(index)] = userid

    for index in user_indices_list:
        # 使用映射找到与给定索引对应的userid
        if index in index_to_user_id_mapping:
            userid = index_to_user_id_mapping[index]
        else:
            raise ValueError(f"No userid found for index {index}")

        # 2. 读取与给定userid对应的embedding
        user_embedding = None
        with open("/content/efficient-xmrec-main/DATA2/proc_data/embeddings_with_userid.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == userid:
                    user_embedding = np.array(list(map(float, parts[1:])))
                    break

        if user_embedding is None:
            raise ValueError(f"User ID {userid} not found in embeddings_with_userid.txt")

        # 3. 计算给定embedding与每个cluster center的距离
        distances = np.linalg.norm(cluster_centers - user_embedding, axis=1)

        # 4. 返回距离最近的cluster center
        nearest_center = cluster_centers[np.argmin(distances)]
        nearest_centers.append(nearest_center)

    nearest_centers_array = np.array(nearest_centers)
    return torch.tensor(nearest_centers_array).to(user_indices_tensor.device)


import json

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def init_market_embedding(model, config):
    model.trainable_market = False
    if config.get("market_aware", False):
        model.market_aware = True
        model.num_markets = config["num_markets"]
        if config.get("embedding_market") is None:
            file_path = '/content/efficient-xmrec-main/DATA2/proc_data/market_aware_reprs.json'
            model.embedding_market = read_json_file(file_path)
            model.trainable_market = True
        else:
            model.embedding_market = config["embedding_market"]
    else:
        model.market_aware = False


def transform_market_aware(model, item_embedding, market_indices):
    # Step 1: Read the index_to_market mapping from the txt file
    index_to_market = {}
    with open('/content/market_to_index.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            market, index = line.strip().split('\t')
            index_to_market[int(index)] = market

    # Convert market_indices tensor to a list
    market_indices_list = market_indices.tolist()

    # Step 2: Transform indices to market names using the index_to_market dictionary
    market_names_list = [index_to_market[idx] for idx in market_indices_list]

    if model.market_aware:
        if model.trainable_market:
            # Get the market embeddings for each market name
            market_embedding_list = [model.embedding_market[market_name] for market_name in market_names_list]
            # Convert the list of embeddings to a tensor
            market_embedding = torch.tensor(market_embedding_list).to(item_embedding.device)
        else:
            # Similar logic for the non-trainable case
            market_embedding_list = [model.embedding_market[market_name] for market_name in market_names_list]
            market_embedding = torch.tensor(market_embedding_list).to(item_embedding.device)

        item_embedding = torch.mul(item_embedding, market_embedding)

    return item_embedding


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False
        init_market_embedding(self, config)

        if config['embedding_user'] is None:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']

        if config['embedding_item'] is None:
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, market_indices=None):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]

        # 获取prototype embedding并与user embedding进行元素级的乘法
        prototype_emb = prototype_embedding(user_indices)
        user_embedding = torch.mul(user_embedding, prototype_emb)

        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]
        item_embedding = transform_market_aware(self, item_embedding, market_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        element_product = element_product.to(torch.float32)
        print(self.affine_output.weight.dtype)
        print(element_product.dtype)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        init_market_embedding(self, config)

    def forward(self, user_indices, item_indices, market_indices=None):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        prototype_emb = prototype_embedding(user_indices)
        user_embedding = torch.mul(user_embedding, prototype_emb)
        item_embedding = transform_market_aware(self, item_embedding, market_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self, args, maml_bool=False):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        gmf_dir, _ = get_model_cid_dir(args, 'gmf')
        resume_checkpoint(gmf_model, model_dir=gmf_dir, device_id=config['device_id'], maml_bool=maml_bool,
                          cuda=config['use_cuda'] is True)
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data

        if self.market_aware:
            assert gmf_model.market_aware
            self.embedding_market.weight.data = gmf_model.embedding_market.weight.data


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        if config.get("market_aware", False):
            assert self.latent_dim_mf == self.latent_dim_mf
            self.latent_dim = self.latent_dim_mf

        init_market_embedding(self, config)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, market_indices=None):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        # 获取prototype embedding并与user embedding进行元素级的乘法
        prototype_emb = prototype_embedding(user_indices)
        user_embedding = torch.mul(user_embedding, prototype_emb)
        item_embedding_mlp = transform_market_aware(self,
                                                                        item_embedding_mlp,
                                                                        market_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        # 获取prototype embedding并与user embedding进行元素级的乘法
        prototype_emb = prototype_embedding(user_indices)
        user_embedding = torch.mul(user_embedding, prototype_emb)
        user_embedding_mf, item_embedding_mf = transform_market_aware(self,
                                                                      item_embedding_mf,
                                                                      market_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self, args, maml_bool=False):
        """Loading weights from trained MLP model & GMF model"""
        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)
        if config['use_cuda'] is True:
            mlp_model.cuda()
        mlp_dir, _ = get_model_cid_dir(args, 'mlp')
        resume_checkpoint(mlp_model, model_dir=mlp_dir, device_id=config['device_id'], maml_bool=maml_bool,
                          cuda=config['use_cuda'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        gmf_dir, _ = get_model_cid_dir(args, 'gmf')
        resume_checkpoint(gmf_model, model_dir=gmf_dir, device_id=config['device_id'], maml_bool=maml_bool,
                          cuda=config['use_cuda'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)

        if self.market_aware:
            assert gmf_model.market_aware
            assert mlp_model.market_aware
            # TODO: what here?
            self.embedding_market.weight.data = mlp_model.embedding_market.weight.data


class NeuMF_MH(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF_MH, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        if config.get("market_aware", False):
            raise ValueError("market_aware forec!")

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # market head (MH) layers
        inout_len = config['layers'][-1] + config['latent_dim_mf']
        # mh_layers_dims = [inout_len, 32, inout_len] #[16,64,32,16,8]
        # mh_layers_dims = [inout_len, inout_len]
        mh_layers_dims = config['mh_layers']
        self.mh_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(mh_layers_dims[:-1], mh_layers_dims[1:])):
            self.mh_layers.append(torch.nn.Linear(in_size, out_size))
        if len(mh_layers_dims) > 0:
            self.affine_output = torch.nn.Linear(in_features=mh_layers_dims[-1], out_features=1)
        else:
            self.affine_output = torch.nn.Linear(in_features=inout_len, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, market_indices=None):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        for idx, _ in enumerate(range(len(self.mh_layers))):
            vector = self.mh_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self, args, maml_bool=False):
        """Loading weights from trained MLP model & GMF model"""
        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)
        if config['use_cuda'] is True:
            mlp_model.cuda()
        mlp_dir, _ = get_model_cid_dir(args, 'mlp')
        resume_checkpoint(mlp_model, model_dir=mlp_dir, device_id=config['device_id'], maml_bool=maml_bool,
                          cuda=config['use_cuda'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        gmf_dir, _ = get_model_cid_dir(args, 'gmf')
        resume_checkpoint(gmf_model, model_dir=gmf_dir, device_id=config['device_id'], maml_bool=maml_bool,
                          cuda=config['use_cuda'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)
