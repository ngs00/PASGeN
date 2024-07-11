import pandas
import numpy
import json
import os
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from rdkit.Chem import MolFromInchi
from scipy import interpolate
from scipy.signal import savgol_filter
from torch_geometric.data import Batch
from util.chem import get_mol_graph


STATE_LABELS = {
    'solid': ['solid', 'Solid', 'SOLID', 'film', 'Film', 'FILM'],
    'liquid': ['liquid', 'Liquid', 'LIQUID', 'melt', 'Melt', 'MELT', 'solution', 'Solution', 'SOLUTION'],
    'gas': ['gas', 'Gas', 'GAS', 'vapor', 'Vapor', 'VAPOR']
}


class IRData:
    def __init__(self, data_id, inchi, mol_graph, state_encoding, wavenumbers, transmittances):
        self.data_id = data_id
        self.inchi = inchi
        self.mol_graph = mol_graph
        self.state_encoding = torch.tensor(state_encoding, dtype=torch.long)
        self.wavenumbers = wavenumbers
        self.transmittances = transmittances
        self.transmittances_savgol = torch.tensor(savgol_filter(transmittances, 128, 3), dtype=torch.float)

    def draw_prediction(self, path_dir, fname, pred):
        plt.figure(figsize=(10, 4))
        plt.grid(linestyle='--')
        plt.plot(self.wavenumbers, self.transmittances_savgol, c='gray', linestyle='--', linewidth=1)
        plt.plot(self.wavenumbers, pred, c='r', linestyle='--', linewidth=1)
        plt.xlim([self.wavenumbers[0], self.wavenumbers[-1]])
        plt.ylim([0, 1])
        plt.title('Data ID: {}\nState: {}\n{}'.format(self.data_id, self.state_encoding.tolist(), self.inchi))

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        plt.savefig(path_dir + '/' + fname, bbox_inches='tight')
        plt.close()


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].mol_graph, self.data[idx].state_encoding, self.data[idx].transmittances_savgol

    @property
    def dim_node_feat(self):
        return self.data[0].mol_graph.x.shape[1]

    @property
    def dim_edge_feat(self):
        return self.data[0].mol_graph.edge_attr.shape[1]

    @property
    def dim_state_encoding(self):
        return self.data[0].state_encoding.shape[0]

    @property
    def len_spect(self):
        return self.data[0].transmittances_savgol.shape[0]

    def split(self, ratio_train, random_seed):
        n_train = int(ratio_train * len(self.data))

        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.random.permutation(len(self.data))
        dataset_train = IRDataset([self.data[idx] for idx in idx_rand[:n_train]])
        dataset_test = IRDataset([self.data[idx] for idx in idx_rand[n_train:]])

        return dataset_train, dataset_test

    def get_k_folds(self, num_folds, random_seed):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.array_split(numpy.random.permutation(len(self.data)), num_folds)
        sub_datasets = list()
        for i in range(0, num_folds):
            sub_datasets.append([self.data[idx] for idx in idx_rand[i]])

        k_folds = list()
        for i in range(0, num_folds):
            dataset_train = IRDataset(list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i+1:])))
            dataset_test = IRDataset(sub_datasets[i])
            k_folds.append([dataset_train, dataset_test])

        return k_folds


def get_state_encoding(str_state):
    label_pos = numpy.full(3, fill_value=1e+9)
    encoding = numpy.zeros(3)

    for i, state in enumerate(STATE_LABELS.keys()):
        for label in STATE_LABELS[state]:
            pos = str_state.find(label)
            if pos > -1:
                label_pos[i] = pos

    if numpy.sum(label_pos) < 3e+9:
        encoding[numpy.argmin(label_pos)] = 1

    return encoding


def load_dataset(path_metadata, path_ir_spect, idx_inchi, idx_ir_id, elem_attrs):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        if pandas.isnull(metadata[i][idx_inchi]):
            continue

        inchi = metadata[i][idx_inchi]
        mol = MolFromInchi(inchi)
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs, add_hydrogen=True)
        if mol_graph is None:
            continue

        if not os.path.isfile(path_ir_spect + '/{}.json'.format(metadata[i][idx_ir_id])):
            continue

        with open(path_ir_spect + '/{}.json'.format(metadata[i][idx_ir_id]), 'r') as f:
            ir_spect = json.load(f)

        if 'state' not in ir_spect.keys():
            continue

        state_encoding = get_state_encoding(ir_spect['state'])
        if numpy.sum(state_encoding) == 0:
            continue

        x = numpy.array(ir_spect['wavenumber'], dtype=float)
        y = numpy.array(ir_spect['transmittance'], dtype=float)

        if x.shape[0] != y.shape[0]:
            continue

        if numpy.min(x) > 600 or numpy.max(x) < 3500:
            continue

        f_interpol = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        wavenumbers = numpy.arange(500, 3501)
        transmittance = f_interpol(wavenumbers)

        data.append(IRData(metadata[i][idx_ir_id], inchi, mol_graph, state_encoding, wavenumbers, transmittance))

    return IRDataset(data)


def collate(batch):
    mol_graphs = list()
    state_encodings = list()
    transmittances = list()

    for b in batch:
        mol_graphs.append(b[0])
        state_encodings.append(torch.argmax(b[1]))
        transmittances.append(b[2].unsqueeze(0))
    mol_graphs = Batch.from_data_list(mol_graphs)
    state_encodings = torch.tensor(state_encodings, dtype=torch.long)
    transmittances = torch.cat(transmittances, dim=0)

    return mol_graphs, state_encodings, transmittances
