import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random

from torch.utils.data import Dataset
from data import utils as du
from data import residue_constants
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from folddof import to_rottrans


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _process_csv_row(processed_file_path):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats['modeled_idx']
    processed_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = np.array(
        random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
    for i,chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')
    return {
        'res_plddt': res_plddt,
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': new_chain_idx,
        'res_idx': new_res_idx,
    }


def _process_csv_row_for_global_pep(processed_file_path, rot_repr_is_q: bool = False):
    processed_feats = du.read_pkl(processed_file_path)
    chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']
    all_chain_idx = np.unique(chain_idx).tolist()
    if len(all_chain_idx) > 1:
        # Randomize chain indices.
        # see: https://github.com/jasonkyuyim/se3_diffusion/issues/10
        # 'This feature and randomization is important when training on complexes.'
        # new_chain_idx = np.zeros_like(res_idx)
        # shuffled_chain_idx = np.array(random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        raise NotImplementedError('TODO.')
    else:
        new_chain_idx = chain_idx
    
    atom_order = [residue_constants.atom_order['N'], residue_constants.atom_order['CA'], residue_constants.atom_order['C'], residue_constants.atom_order['O']]
    
    for idx, chain_id in enumerate(all_chain_idx):
        chain_mask = (processed_feats['chain_index'] == chain_id)
        obs_mask = processed_feats['bb_mask'].astype(np.bool_)
        std_mask = processed_feats['aatype'] < 20
        if not std_mask[chain_mask].any(): continue
        use_mask = chain_mask & obs_mask & std_mask
        obs_author_residue_number = res_idx[use_mask]

        assert std_mask[chain_mask & obs_mask].all(), f"Tempoary discard chains with non-standard amino acid residues. (from: {processed_file_path})"
        assert np.unique(obs_author_residue_number).shape[0] == obs_author_residue_number.shape[0], f"Tempoary discard chains with author insertion codes. TODO: regenerate pkl and use residue_number instead of author_residue_number for `residue_index`! (from: {processed_file_path})"
        assert tuple(np.unique(obs_author_residue_number[1:] - obs_author_residue_number[:-1]).tolist()) == (1,), f"Tempoary discard chains with potential missing residues. TODO: regenerate pkl and use residue_number instead of author_residue_number for `residue_index`! (from: {processed_file_path})"
        
        # Re-number residue indices for each chain such that it starts from 1.
        new_res_idx = np.arange(1, obs_author_residue_number.shape[0]+2) # obs_author_residue_number - obs_author_residue_number[0] + 1
        # aatype = torch.tensor(processed_feats['aatype'][use_mask]).long()
        
        bb_coords = torch.from_numpy(processed_feats['atom_positions'][use_mask][:, atom_order]).to(dtype=torch.float)
        bb_mask = torch.from_numpy(processed_feats['atom_mask'][use_mask][:, atom_order]).to(dtype=torch.bool)
        rotmats_1, trans_1, loc_ca_ia1_wrt_n_ia1_1, pep_mask_1, loc_ca_ia1_wrt_n_ia1_mask_1 = to_rottrans(bb_coords.transpose(0, 1), bb_mask.transpose(0, 1), rot_repr_is_q=rot_repr_is_q)
        #rotmats_1 = rotmats_1.numpy(); trans_1 = trans_1.numpy(); pep_mask_1 = pep_mask_1.numpy()
        #res_plddt = processed_feats['b_factors'][use_mask][:, 1]

        assert pep_mask_1.all(), f"Tempoary discard chains with missing pep frames. (from: {processed_file_path})"
        assert loc_ca_ia1_wrt_n_ia1_mask_1.all(), f"Tempoary discard chains with missing loc_ca_ia1_wrt_n_ia1. (from: {processed_file_path})"
        
        # Shuffle chain_index
        # ...

        break
    
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')
    
    return {
        #'res_plddt': res_plddt,
        #'aatype': aatype,
        'rotmats_1': rotmats_1, # L+1
        'trans_1': trans_1, # L+1
        'res_mask': pep_mask_1.int(), # L+1
        '_res_mask': pep_mask_1[1:].int(), # L
        'loc_ca_ia1_wrt_n_ia1_1': loc_ca_ia1_wrt_n_ia1_1, # L
        #'is_trans': PeptideUnitFrame.whether_is_trans(loc_ca_ia1_wrt_n_ia1_1),
        #'chain_idx': new_chain_idx,
        'res_idx': new_res_idx, # L+1
    }


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            bb_repr,
            rot_repr_is_q,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._bb_repr = bb_repr
        self._rot_repr_is_q = rot_repr_is_q
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123
            )
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        if self._bb_repr == 'original':
            processed_row = _process_csv_row(path)
        else:
            processed_row = _process_csv_row_for_global_pep(path, rot_repr_is_q=self._rot_repr_is_q)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row
    
    def _sample_scaffold_mask(self, batch, rng):
        trans_1 = batch['trans_1']
        num_res = trans_1.shape[0]
        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(
            low=min_motif_size,
            high=max_motif_size
        )

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(
                    low=1,
                    high=max_motif_size,
                    size=(num_motifs,)
                )
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Sample start location of each motif.
        seed_residues = rng.integers(
            low=0,
            high=num_res-1,
            size=(len(motif_lengths),)
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            motif_mask[motif_seed:min(motif_seed+motif_len, num_res)] = 1.0
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * batch['res_mask']
    
    def setup_inpainting(self, feats, rng):
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            diffuse_mask = diffuse_mask * feats['plddt_mask']
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats['diffuse_mask'] = diffuse_mask

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':
            if self._dataset_cfg.inpainting_percent < random.random():
                feats['diffuse_mask'] = torch.ones_like(feats['res_mask'])    
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats['diffuse_mask']
                trans_1 = feats['trans_1']
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats['trans_1'] = trans_1
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['diffuse_mask'] = feats['diffuse_mask'].int()

        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
        return feats


class ScopeDataset(BaseDataset):

    def _filter_metadata(self, raw_csv):
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv,
            filter_cfg.min_num_res,
            filter_cfg.max_num_res
        ).reset_index(drop=True)
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv


class PdbDataset(BaseDataset):

    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)

        self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0
        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[
            raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
        data_csv = data_csv[
            data_csv.num_chains.isin(filter_cfg.num_chains)]
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        return data_csv
