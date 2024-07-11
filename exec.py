import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.chem import load_elem_attrs
from util.data import load_dataset, collate
from method.mol_encoder import MPNN
from method.pasgen import PASGeN


model_name = 'pasgen'
random_seed = 0
num_folds = 5
batch_size = 32
dim_emb = 128
dim_state_emb = 32
num_epochs = 500


elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset = load_dataset(path_metadata='../../data/chem_data/ir/nist/refined/metadata.xlsx',
                       path_ir_spect='../../data/chem_data/ir/nist/refined',
                       idx_inchi=3, idx_ir_id=0, elem_attrs=elem_attrs)
k_folds = dataset.get_k_folds(num_folds, random_seed=random_seed)

for k in range(0, num_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)
    targets_test = torch.vstack([d.transmittances_savgol for d in dataset_test.data])

    mol_encoder = MPNN(dataset_train.dim_node_feat, dataset_train.dim_edge_feat + dim_state_emb,
                       dim_hidden=128, dim_out=dim_emb)
    model = PASGeN(mol_encoder, dataset_train.dim_state_encoding, dim_emb, dim_state_emb,
                   dataset_train.len_spect).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-6)

    for epoch in range(0, num_epochs):
        loss_train = model.fit(loader_train, optimizer)
        print('Epoch [{}/{}]\tTraining loss: {:.4f}'.format(epoch + 1, num_epochs, loss_train))

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), 'save/model_{}_{}.pt'.format(model_name, k))

    IDX_TO_STATE = {0: 'SOLID', 1: 'LIQUID', 2: 'GAS'}
    dirs = list()
    preds_test = model.predict(loader_test)
    for i in tqdm(range(0, preds_test.shape[0])):
        data = dataset_test.data[i]
        path_dir = 'save/preds_ir_spectrum_{}_{}/{}'.format(model_name, k, data.data_id.split('_')[0])
        state_label = IDX_TO_STATE[torch.argmax(data.state_encoding).item()]
        fname = '{}_{}.png'.format(data.data_id, state_label)
        data.draw_prediction(path_dir, fname, preds_test[i].numpy())

        ir_pred = {'x': data.wavenumbers.tolist(),
                   'y_true': data.transmittances_savgol.tolist(),
                   'y_pred': preds_test[i].tolist()}
        with open(path_dir + '/{}_{}.json'.format(data.data_id, state_label), 'w') as file:
            json.dump(ir_pred, file)

        dirs.append(path_dir)
