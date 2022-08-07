import os
import numpy as np
import torch

from benchmarkutils import separate_data_by_state
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, SAGPooling, global_mean_pool, global_max_pool, EdgePooling

# from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class TrafficGNN(nn.Module):

    def __init__(self, d_in, h_1, h_2, mode='state'):
        super(TrafficGNN, self).__init__()
        self.hidden_dim = 12
        self.input_num = 14
        self.in_channel = 32
        self.inner_edge = torch.tensor([[0, 0, 0, 0, 11, 12, 13, 5, 6, 2, 8, 6, 9, 8, 5, 10],
                                        [1, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]]).to(device)

        self.inner_sub_edge = torch.tensor([[0, 0, 0, 0, 2],
                                            [1, 2, 3, 4, 3]]).to(device)

        self.reverse_edge = torch.tensor([[1, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                                          [0, 0, 0, 0, 11, 12, 13, 5, 6, 2, 8, 6, 9, 8, 5, 10]]).to(device)

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=12, num_heads=3, dropout=0.5)
        # self.embedding_attention = AttentionBlock(self.input_num, self.hidden_dim)
        self.inner_gcn = GCNConv(self.hidden_dim, self.hidden_dim, node_dim=1)
        self.fc = nn.Linear(36, 12)
        self.fc_f = nn.Linear(12, 1)
        self.fc_q = nn.Linear(12, 1)
        self.mode = mode

    def forward(self, x):

        x = x.view(x.shape[0], 14, x.shape[1] // 14)
        x = x.to(device)
        x_embedding, _ = self.gru(x)
        x_trans_embedding = x_embedding.transpose(0, 1)
        # attention
        x_trans_att_vector, _ = self.multihead_attn(x_trans_embedding, x_trans_embedding, x_trans_embedding)
        x_att_vector = x_trans_att_vector.transpose(0, 1)
        inner_graph_embedding = self.inner_gcn(x_att_vector, self.inner_edge)
        batch = torch.zeros(12).long().to(device)
        inner_graph_embedding = inner_graph_embedding.transpose(0, 2)
        inner_graph_embedding = torch.add(global_max_pool(inner_graph_embedding, batch),
                                          global_mean_pool(inner_graph_embedding, batch))
        inner_graph_embedding = inner_graph_embedding.transpose(0, 2)
        inner_subgraph_pooling_embedding = self.inner_gcn(inner_graph_embedding.expand(-1, -1, 12), self.inner_sub_edge)

        # reversed  convolution
        reverse_vectors = self.inner_gcn(torch.fliplr(x_att_vector), self.reverse_edge)  # [32, 14, 12]
        # fusion
        fusion_vec = torch.cat((x_att_vector, inner_subgraph_pooling_embedding, reverse_vectors), dim=-1)
        # print(fusion_vec[:, :1, :].shape)
        fusion_vec = torch.flatten(fusion_vec[:, :1, :], start_dim=1, end_dim=-1)
        # print(fusion_vec.shape)
        x = F.dropout(fusion_vec, training=self.training)
        x = F.relu(x)
        x = self.fc(x)

        f = self.fc_f(x)
        q = self.fc_q(x)
        return f, q

    def fit_state(self,
                  X,
                  y_data,
                  y_state,
                  criterion=nn.L1Loss(),
                  n_steps=10000,
                  batch_size=32,
                  lr=1e-3,
                  momentum=0.9,
                  weight_decay=0,
                  verbose=True):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        X_f, y_f, X_q, y_q = separate_data_by_state(X, y_data, y_state)
        f_indices = np.arange(X_f.shape[0])
        q_indices = np.arange(X_q.shape[0])
        loss_val = 0

        for n in range(n_steps):
            optimizer.zero_grad()
            if n % 2:
                batch_idx = np.random.choice(q_indices, batch_size)
                X_batch = torch.from_numpy(X_q[batch_idx]).float().to(device)
                y_batch = torch.from_numpy(y_q[batch_idx]).unsqueeze(1).float().to(device)
                _, y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            else:
                batch_idx = np.random.choice(f_indices, batch_size)
                X_batch = torch.from_numpy(X_f[batch_idx]).float().to(device)
                y_batch = torch.from_numpy(y_f[batch_idx]).unsqueeze(1).float().to(device)
                y_pred, _ = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            loss_val += loss.item()
            if verbose and (n + 1) % 1000 == 0:
                print('loss for steps {}-{}: {}'.format(n - 999, n + 1, loss_val / 1000))
                loss_val = 0
            if n == int(n_steps / 2):
                optimizer = optim.SGD(self.parameters(), lr=lr / 10, momentum=momentum, weight_decay=weight_decay)
        return

    def fit_full(self,
                 X,
                 y_data,
                 criterion=nn.L1Loss(),
                 n_steps=10000,
                 batch_size=32,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=0,
                 verbose=True):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        indices = np.arange(X.shape[0])
        loss_val = 0

        for n in range(n_steps):
            optimizer.zero_grad()
            batch_idx = np.random.choice(indices, batch_size)
            X_batch = torch.from_numpy(X[batch_idx]).float().to(device)
            y_batch = torch.from_numpy(y_data[batch_idx]).unsqueeze(1).float().to(device)
            y_pred, _ = self.forward(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            if verbose and (n + 1) % 1000 == 0:
                print('loss for steps {}-{}: {}'.format(n - 999, n + 1, loss_val / 1000))
                loss_val = 0
            if n == int(n_steps / 2):
                optimizer = optim.SGD(self.parameters(), lr=lr / 10, momentum=momentum, weight_decay=weight_decay)

    def predict(self, X, y_state):
        print("this is test set:")
        predictions = np.zeros(len(y_state))
        if self.mode == 'state':
            with torch.no_grad():
                f_preds, q_preds = self.forward(torch.from_numpy(X).float().to(device))
            f_indices = y_state == 0
            q_indices = y_state == 1
            predictions[f_indices] = (f_preds.cpu().squeeze(1).numpy())[f_indices]
            predictions[q_indices] = (q_preds.cpu().squeeze(1).numpy())[q_indices]
        else:
            with torch.no_grad():
                preds, _ = self.forward(torch.from_numpy(X).float().to(device))
            predictions = preds.cpu().squeeze(1).numpy()
        return predictions