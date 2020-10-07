#!/usr/bin/env python
# coding: utf-8

from gluonts.dataset.common import ListDataset


#def ListDatasetX(dataset, freq, one_dim_target, simple=False):
def ListDatasetX(dataset, freq, one_dim_target, simple=True):
    """
    wrapper for ListDataSet

    """

    _dataset = []
    if simple:
        #copy only target and 'start'
        for rec in dataset:
            _dataset.append({'target':rec['target'],'start':rec['start'],
                    'feat_static_cat':rec['feat_static_cat']})
    else:
        _dataset = dataset

    return ListDataset(_dataset, freq = freq, one_dim_target = one_dim_target)