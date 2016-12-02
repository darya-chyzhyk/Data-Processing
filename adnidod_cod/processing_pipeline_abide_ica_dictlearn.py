"""Processing pipeline example for resting state fMRI datasets
"""
import os
import itertools
import numpy as np


def draw_predictions(imgs=None, labels=None, index=None,
                     train_index=None, test_index=None,
                     scoring='roc_auc', models=None, atlases=None,
                     masker=None, connectomes=None,
                     confounds=None, confounds_mask_img=None,
                     connectome_regress_confounds=None):
    """
    """
    learn_brain_regions = LearnBrainRegions(
        model=models,
        atlases=atlases,
        masker=masker,
        connectome_convert=True,
        connectome_measure=connectomes,
        min_region_size=1500,
        connectome_confounds=connectome_regress_confounds,
        n_comp=40,
        compute_confounds='compcor_10',
        compute_confounds_mask_img=confounds_mask_img,
        compute_not_mask_confounds=None,
        verbose=2)
    print("Processing index={0}".format(index))

    train_data = [imgs[i] for i in train_index]

    # Fit the model to learn brain regions on the data
    learn_brain_regions.fit(train_data)

    # Tranform into subjects timeseries signals and connectomes
    # from a learned brain regions on training data. Now timeseries
    # and connectomes are extracted on all images
    learn_brain_regions.transform(imgs, confounds=confounds)

    # classification scores
    learn_brain_regions.classify(labels, cv=[(train_index, test_index)],
                                 scoring='roc_auc')
    print(learn_brain_regions.scores_)

    return learn_brain_regions

###########################################################################
# Data
# ----
# Load the datasets

import load_datasets

data_path = '/media/kr245263/SAMSUNG/'
datasets_ = ['abide']
data_store = dict()
cache = dict()
for path, dataset in zip(itertools.repeat(data_path), datasets_):
    print("Loading %s datasets from %s path" % (dataset, path))
    data_store[dataset] = load_datasets.fetch(dataset_name=dataset,
                                              data_path=path)
    cache[dataset] = os.path.join(data_path, ('data_processing_' + dataset +
                                              '_analysis'))

# Data to process
name = 'abide'
# class type for each subject is different
class_type = 'DX_GROUP'
cache_path = cache[name]

from joblib import Memory, Parallel, delayed
mem = Memory(cachedir=cache_path)

func_imgs = data_store[name].func_preproc
phenotypic = data_store[name].phenotypic
connectome_regress_confounds = None

from nilearn_utils import data_info

shape, affine, _ = data_info(func_imgs[0])
###########################################################################
# Masker
# ------
# Masking the data

from nilearn import datasets

# Fetch grey matter mask from nilearn shipped with ICBM templates
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=gm_mask, target_shape=shape,
                          target_affine=affine, smoothing_fwhm=6.,
                          standardize=True, detrend=True, mask_strategy='epi',
                          memory=mem, memory_level=2, n_jobs=10,
                          verbose=10)

##############################################################################
# Cross Validator
# ---------------

from sklearn.cross_validation import StratifiedShuffleSplit

n_iter = 100
classes = phenotypic[class_type]
_, labels = np.unique(classes, return_inverse=True)
cv = StratifiedShuffleSplit(classes, n_iter=n_iter,
                            test_size=0.25, random_state=0)
##############################################################################
# Fetching Predefined atlases
# ---------------------------
from regions_definition import Atlases

# Predefined atlas object
atlases = Atlases(
    aal_version='SPM12', harvard_atlas_name='cort-maxprob-thr25-2mm',
    symmetric_split=True, basc_version='asym', basc_scale='scale122',
    memory=mem, memory_level=1)

# atlas without region extraction in index 0 and with in index 1
aal_atlas_img = atlases.fetch_aal(region_extraction=False)

ho_atlas_img = atlases.fetch_harvard_oxford(region_extraction=False)

basc_atlas_img = atlases.fetch_basc(region_extraction=False)

# Pool all atlases to dictionary like with name assgined to each one
atlases_ = dict()
atlases_['aal_spm12'] = aal_atlas_img
atlases_['ho_cort_symm_split'] = ho_atlas_img
atlases_['basc_scale122'] = basc_atlas_img

###############################################################################
# Functional Connectivity Analysis model
# ---------------------------------------
from model import LearnBrainRegions

connectomes = ['correlation', 'partial correlation', 'tangent']

############################################################################
# Gather results - Data structure
from _gather_results import _set_default_dict_with_labels, _append_results

columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'n_regions', 'smoothing_fwhm', 'dataset', 'compcor_10', 'motion_regress',
           'n_components', 'n_clusters', 'connectome_regress', 'scoring']
results = _set_default_dict_with_labels(columns)
print(results)

##############################################################################
# Run the analysis now
# --------------------
folder_name = name + str(n_iter) + 'smooth_6_ica_dictlearn'
for model in ['ica', 'dictlearn']:
    for index, (train_index, test_index) in enumerate(cv):
        all_results = draw_predictions(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            confounds_mask_img=gm_mask,
            connectome_regress_confounds=connectome_regress_confounds)
        print(index)
        # Dump the results
        for model_ in all_results.models_:
            save_path = os.path.join(folder_name, model_, str(index))
            print(save_path)
            if not os.path.exists(save_path):
                print("Making directory {0}".format(save_path))
                os.makedirs(save_path)
            # parcellations
            direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
            atlas_parcel = all_results.parcellations_[model_]
            atlas_parcel.to_filename(direc_parcel)
            print(direc_parcel)
            # regions of interest
            direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
            atlas_roi = all_results.rois_[model_]
            atlas_roi.to_filename(direc_roi)
            print(direc_roi)
        results = _append_results(results, all_results, index)

#################################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------
import pandas as pd

results_csv = pd.DataFrame(results)
results_csv.to_csv(folder_name + '.csv')
print("Done................")
