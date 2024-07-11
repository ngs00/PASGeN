import torch.nn
from torch.nn.functional import leaky_relu, normalize
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import global_add_pool
from method.transformer import TransformerBlock


WINDOW_SIZE = 24


def pearson_corr(x1, x2):
    _x1 = x1 - torch.mean(x1, dim=1, keepdim=True)
    _x2 = x2 - torch.mean(x2, dim=1, keepdim=True)
    std1 = torch.sqrt(torch.sum(_x1**2, dim=1))
    std2 = torch.sqrt(torch.sum(_x2**2, dim=1))

    return torch.sum(_x1 * _x2, dim=1) / (std1 * std2 + 1e-10)


class SpectrumDecoder(torch.nn.Module):
    def __init__(self, dim_emb, len_spect):
        super(SpectrumDecoder, self).__init__()
        self.dim_emb = dim_emb
        self.len_spect = len_spect

        self.transformer = TransformerBlock(WINDOW_SIZE, dim_head=32, num_heads=4, len_spect=362)
        self.pred_layer = torch.nn.Sequential(
            torch.nn.Linear(WINDOW_SIZE + dim_emb, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, len_spect),
        )
        self.emb_eng = torch.nn.Embedding(self.len_spect, dim_emb)
        self.fc1 = torch.nn.Linear(125, 32)
        self.fc2 = torch.nn.Linear(dim_emb + 32, 1)

    def forward(self, x):
        token_eng = torch.arange(0, self.len_spect, dtype=torch.long).cuda()
        z_eng = self.emb_eng(token_eng).unsqueeze(0).repeat(x.shape[0], 1, 1)
        _x = x.unsqueeze(1).repeat(1, self.len_spect, 1)
        z = leaky_relu(torch.sum(_x * z_eng, dim=2))

        z = torch.cat([torch.zeros(z.shape[0], 6).cuda(), z, torch.zeros(z.shape[0], 6).cuda()], dim=1)
        z = z.unfold(dimension=1, size=WINDOW_SIZE, step=1)[:, ::WINDOW_SIZE, :]

        # Local embedding.
        z = self.transformer(z)

        # Local-global embedding.
        z = torch.cat([z, x.unsqueeze(1).repeat(1, z.shape[1], 1)], dim=2)
        h = leaky_relu(self.fc1(leaky_relu(self.pred_layer(z)).swapaxes(1, 2)))
        h = torch.cat([h, z_eng], dim=2)
        out = self.fc2(h).squeeze(2)

        return out, z


class PASGeN(torch.nn.Module):
    def __init__(self, mol_encoder, dim_state_encoding, dim_emb, dim_state, len_spect):
        super(PASGeN, self).__init__()
        self.mol_encoder = mol_encoder
        self.state_encoder = torch.nn.Embedding(dim_state_encoding, dim_state)
        self.spect_decoder = SpectrumDecoder(2 * dim_emb, len_spect)
        self.fc_attn = torch.nn.Sequential(
            torch.nn.Linear(dim_emb + dim_state, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, mol_graph, state):
        z_state = self.state_encoder(state)
        edge_attr = torch.cat([mol_graph.edge_attr, z_state[mol_graph.batch[mol_graph.edge_index[0]]]], dim=1)
        z_mol, z_atom = self.mol_encoder(mol_graph.x, mol_graph.edge_index, edge_attr, mol_graph.batch)

        attn = self.fc_attn(torch.cat([z_atom, z_state[mol_graph.batch]], dim=1))
        attn = softmax(attn, mol_graph.batch)
        z_mol_state = global_add_pool(attn * z_atom, mol_graph.batch)
        out, z = self.spect_decoder(torch.cat([z_mol, z_mol_state], dim=1))

        return out, z

    def fit(self, data_loader, optimizer):
        sum_losses = 0

        self.train()
        for g, s, y in data_loader:
            preds, _ = self(g.cuda(), s.cuda())
            loss = torch.mean((preds - y.cuda())**2)
            loss += torch.mean(-pearson_corr(preds, y.cuda()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()

        with torch.no_grad():
            for g, s, _ in data_loader:
                preds, _ = self(g.cuda(), s.cuda())
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0).cpu()

    def get_embs(self, data_loader):
        list_embs = list()

        with torch.no_grad():
            for g, s, _ in data_loader:
                _, z = self(g.cuda(), s.cuda())
                list_embs.append(torch.mean(z, dim=2))

        return torch.cat(list_embs, dim=0).cpu()


def get_pasgen_embs(pasgen, g, s):
    with torch.no_grad():
        _, z = pasgen(g.cuda(), s.cuda())

    return normalize(torch.mean(z, dim=2), p=2, dim=1)
