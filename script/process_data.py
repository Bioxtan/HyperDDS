
import pandas as pd
import deepchem as dc
from rdkit import Chem
import numpy as np
from utils import *
import json




def getData(drug_synergy_file,cell_features_file):
    drug_synergy_file = drug_synergy_file
    cell_features_file = json.loads(open(cell_features_file,'r').read())

    synergy = pd.read_csv(drug_synergy_file,sep=',',header=0)
    drugs = set(synergy['drug1']).union(set(synergy['drug2']))
    drug_data = pd.DataFrame()
    drug_smiles_maccs = []
    featurizer = dc.feat.ConvMolFeaturizer()

    for smiles in drugs:
        mol  = Chem.MolFromSmiles(smiles)
        mol_f = featurizer.featurize(mol)
        drug_data[smiles] = [mol_f[0].get_atom_features(),mol_f[0].get_adjacency_list()]
        drug_smiles_maccs.append(get_MACCS(smiles))
    drug_num = len(drug_data.keys())
    d_map = dict(zip(drug_data.keys(),range(drug_num)))
    drug_feature = drug_feature_extract(drug_data)

    cell_line_set = set(synergy['cell'])
    filtered_features = {k: v for k, v in cell_features_file.items() if k in cell_line_set}
    cell_feature = pd.DataFrame.from_dict(filtered_features, orient='index')
    cell_num = len(cell_feature.index)
    c_map = dict(zip(cell_feature.index,range(drug_num,drug_num + cell_num)))
    cline_fea = np.array(cell_feature,dtype = 'float32')

    hypersynergy = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], int(row[3])] for index, row in
               synergy.iterrows() if (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and
                                           str(row[2]) in cell_feature.index)]
    return drug_feature,cline_fea,hypersynergy