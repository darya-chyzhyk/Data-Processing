""" Gather all the pipeline results
"""
import pandas as pd
import numpy as np
import warnings
import collections

from nilearn.image import load_img


def _set_default_dict_with_labels(columns):
    """Create a default dictionary with same column names as
    in Pandas Data frame.

    Parameters
    ----------
    columns : array-like
        Column labels to use for resulting frame.
        Example: columns=['a', 'b', 'c', 'd', 'e']

    Returns
    -------
    results : dict
        Setting a dictionary assignments with keys
        from the input pandas data frame.
    """
    results = dict()
    if isinstance(columns, collections.Iterable):
        for label in columns:
            results.setdefault(label, [])
    else:
        warnings.warn("columns is expected as Iterable type, either "
                      "as a list or dict. Returning empty")
        results = []

    return results


def _append_results(results, model, iteration, dataset_name, dimension):
    """Gather results from a model which has attributes.

    Parameters
    ----------
    results : dict
        Should contain columns with empty array list for appending
        all the cross validation results for each iteration in
        cross_val_score
        {'atlas': [], 'classifier': [], 'measure': [], 'scores': []}
    model : object, instance of LearnBrainRegions

    Return
    ------
    results : dictionary
    """
    for atlas in model.models_:
        for measure in ['correlation', 'partial correlation', 'tangent']:
            for classifier in ['svc_l1', 'svc_l2', 'ridge']:
                results['iter_shuffle_split'].append(iteration)
                results['atlas'].append(atlas)
                results['measure'].append(measure)
                results['classifier'].append(classifier)
                results['scoring'].append('roc_auc')
                results['dataset'].append(dataset_name)
                results['compcor_10'].append('yes')
                results['motion_regress'].append('no')
                results['smoothing_fwhm'].append(6.)
                results['connectome_regress'].append('no')
		results['confounds'].append('yes')
		results['compute_confounds'].append('no')
                score = model.scores_[atlas][measure][classifier]
                results['scores'].append(score)

                print("Loading rois_of {0} to regions".format(atlas))
                if atlas == 'aal_spm12':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(117)
                elif atlas == 'ho_cort_symm_split':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(96)
                elif atlas == 'yeo':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(17)
                elif atlas == 'basc_scale122':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(122)
                elif atlas == 'basc_scale122_sym':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(122)
                elif atlas == 'basc_scale036':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(36)
                elif atlas == 'destrieux':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(76)
                elif atlas == 'craddok':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(950)
                elif atlas == 'smith_bm70':
                    results['dimensionality'].append('NA')
                    results['n_regions'].append(70)
                elif atlas == 'ica':
                    results['dimensionality'].append(dimension)
                    rois_img = load_img(model.rois_[atlas])
                    if len(rois_img.shape) > 3:
                        results['n_regions'].append(rois_img.shape[3])
                elif atlas == 'dictlearn':
                    results['dimensionality'].append(dimension)
                    rois_img = load_img(model.rois_[atlas])
                    if len(rois_img.shape) > 3:
                        results['n_regions'].append(rois_img.shape[3])
                elif atlas == 'kmeans':
                    results['dimensionality'].append(dimension)
                    results['n_regions'].append(120)
                elif atlas == 'ward':
                    results['dimensionality'].append(dimension)
                    results['n_regions'].append(120)
                else:
                    results['dimensionality'].append(dimension)
                    results['n_regions'].append('NA')

    return results

