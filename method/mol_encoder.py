import torch
from torch_scatter import scatter_add
from torch.nn.functional import leaky_relu, normalize
from torch_geometric.nn.conv import NNConv, GATv2Conv, TransformerConv, CGConv
from torch_geometric.nn.glob import global_add_pool
from copy import deepcopy


class MPNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(MPNN, self).__init__()
        self.dim_hidden = dim_hidden
        self.nfc = torch.nn.Linear(dim_node_feat, self.dim_hidden)
        self.efc1 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, self.dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.dim_hidden, self.dim_hidden * self.dim_hidden))
        self.gc1 = NNConv(self.dim_hidden, self.dim_hidden, self.efc1)
        self.efc2 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, self.dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.dim_hidden, self.dim_hidden * self.dim_hidden))
        self.gc2 = NNConv(self.dim_hidden, self.dim_hidden, self.efc2)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.dropout(leaky_relu(self.nfc(x)))
        h = self.dropout(leaky_relu(self.gc1(h, edge_index, edge_attr)))
        atom_embs = self.dropout(leaky_relu(self.gc2(h, edge_index, edge_attr)))
        h = normalize(global_add_pool(atom_embs, batch), p=2, dim=1)
        out = self.fc(h)

        return out, atom_embs


class GAT(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(GAT, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = GATv2Conv(in_channels=dim_node_feat, edge_dim=dim_edge_feat, heads=4,
                             concat=False, out_channels=self.dim_hidden)
        self.gc2 = GATv2Conv(in_channels=self.dim_hidden, edge_dim=dim_edge_feat, heads=4,
                             concat=False, out_channels=self.dim_hidden)
        self.fc1 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc2 = torch.nn.Linear(self.dim_hidden, dim_out)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.dropout(leaky_relu(self.gc1(x, edge_index, edge_attr)))
        atom_embs = self.dropout(leaky_relu(self.gc2(h, edge_index, edge_attr)))
        h = normalize(global_add_pool(atom_embs, batch), p=2, dim=1)
        h = leaky_relu(self.fc1(h))
        out = self.fc2(h)

        return out, atom_embs


class UniMP(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(UniMP, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = TransformerConv(dim_node_feat, self.dim_hidden, edge_dim=dim_edge_feat)
        self.gc2 = TransformerConv(self.dim_hidden, self.dim_hidden, edge_dim=dim_edge_feat)
        self.fc1 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc2 = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, x, edge_index, edge_attr, batch):
        h = leaky_relu(self.gc1(x, edge_index, edge_attr))
        atom_embs = leaky_relu(self.gc2(h, edge_index, edge_attr))
        h = normalize(global_add_pool(atom_embs, batch), p=2, dim=1)
        h = leaky_relu(self.fc1(h))
        out = self.fc2(h)

        return out, atom_embs


class CGCNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(CGCNN, self).__init__()
        self.dim_hidden = dim_hidden
        self.fc_node = torch.nn.Linear(dim_node_feat, self.dim_hidden)
        self.gc1 = CGConv(self.dim_hidden, dim=dim_edge_feat)
        self.gc2 = CGConv(self.dim_hidden, dim=dim_edge_feat)
        self.fc1 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc2 = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, x, edge_index, edge_attr, batch):
        h = leaky_relu(self.fc_node(x))
        h = leaky_relu(self.gc1(h, edge_index, edge_attr))
        atom_embs = leaky_relu(self.gc2(h, edge_index, edge_attr))
        h = normalize(global_add_pool(atom_embs, batch), p=2, dim=1)
        h = leaky_relu(self.fc1(h))
        out = self.fc2(h)

        return out, atom_embs


class DirectedMPNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(DirectedMPNN, self).__init__()
        self.dim_hidden = dim_hidden
        self.fc_init = torch.nn.Linear(dim_edge_feat, dim_hidden)

        self.fc_hidden1 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_hidden2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_hidden3 = torch.nn.Linear(dim_hidden, dim_hidden)

        self.fc_mol = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_out = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, x, concat_feats, srt_concat_batch, end_concat_batch, num_concat_feats, batch):
        concat_feats = concat_feats
        _srt_concat_batch = deepcopy(srt_concat_batch)
        _end_concat_batch = deepcopy(end_concat_batch)

        h_0 = leaky_relu(self.fc_init(concat_feats))

        m_1 = scatter_add(h_0, _end_concat_batch, dim=0)[_srt_concat_batch] - h_0[_end_concat_batch]
        h_1 = leaky_relu(h_0 + self.fc_hidden1(m_1))

        m_2 = scatter_add(h_1, _end_concat_batch, dim=0)[_srt_concat_batch] - h_1[_end_concat_batch]
        h_2 = leaky_relu(h_1 + self.fc_hidden1(m_2))

        m_3 = scatter_add(h_2, _end_concat_batch, dim=0)[_srt_concat_batch] - h_2[_end_concat_batch]
        h_3 = leaky_relu(h_2 + self.fc_hidden1(m_3))

        h_node = scatter_add(h_3, _srt_concat_batch, dim=0)
        atom_embs = leaky_relu(self.fc_mol(h_node))

        hg = global_add_pool(normalize(atom_embs, p=2, dim=1), batch)
        out = self.fc_out(hg)

        return out, atom_embs
