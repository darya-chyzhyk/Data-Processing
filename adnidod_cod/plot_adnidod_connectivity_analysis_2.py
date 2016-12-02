"""Processing pipeline example for resting state fMRI datasets
"""
import os
import itertools
import numpy as np


#def _append_results(results, model):
#    """Gather results from a model which has attributes.
#
#    Parameters
#    ----------
#    results : dict
#        Should contain columns with empty array list for appending
#        all the cross validation results for each iteration in
#        cross_val_score
#        {'atlas': [], 'classifier': [], 'measure': [], 'scores': []}
#
#    model : object, instance of LearnBrainRegions
#
#    Return
#    ------
#    results : dictionary
#    """
#    for atlas in model.models_:
#        for measure in ['correlation', 'partial correlation', 'tangent']:
#            for classifier in ['svc_l1', 'svc_l2', 'ridge']:
#                results['atlas'].append(atlas)
#                results['measure'].append(measure)
#                results['classifier'].append(classifier)
#                score = model.scores_[atlas][measure][classifier]
#                results['scores'].append(score)
#    return results


def draw_predictions(imgs=None, labels=None, index=None,
                     train_index=None, test_index=None,
                     scoring='roc_auc', models=None, atlases=None,
                     masker=None, connectomes=None,
                     connectome_regress_confounds=None, 
                     compute_confounds = None, confounds=None): 
                         #index=None,
    """
    """
    learn_brain_regions = LearnBrainRegions(
        model=models,
        atlases=atlases,
        masker=masker,
        connectome_convert=True,
        connectome_measure=connectomes,
        connectome_confounds=connectome_regress_confounds, # None
        n_comp=40,
        n_parcels = 120, #n_clusters=120,
        compute_confounds = compute_confounds, #None,#'compcor_10', #None
        compute_not_mask_confounds=None,
        compute_confounds_mask_img= None, #gm_mask,
        verbose=2)
    print("Processing index={0}".format(index))

    train_data = [imgs[i] for i in train_index]

    # Fit the model to learn brain regions on the data
    learn_brain_regions.fit(train_data)

    # Tranform into subjects timeseries signals and connectomes
    # from a learned brain regions on training data. Now timeseries
    # and connectomes are extracted on all images
    learn_brain_regions.transform(imgs, confounds=confounds) #  use my confounds

    # classification scores
    # By default it used two classifiers, LinearSVC ('l1', 'l2') and Ridge
    # Not so good documentation and implementation here according to me
    learn_brain_regions.classify(labels, cv=[(train_index, test_index)],
                                 scoring='roc_auc')
    print(learn_brain_regions.scores_)

    return learn_brain_regions

###########################################################################
# Data
# ----
# ADNIDOD data
import pandas as pd

dir_fmri_list = '/volatile/darya/Documents/experiments/analysis'

df_demog = pd.read_csv(os.path.join(dir_fmri_list, 'adnidod_demographic.csv'))

df_fmri_conf_list = pd.read_csv(os.path.join(dir_fmri_list, 'fmri_conf_list.csv')) 
#df_fmri_conf_demog = pd.merge(df_fmri_conf_list, df_demog, on='ID_scan')

#func_imgs = list(df_fmri_conf_demog['fmri_path'])


###### in case of conf calculation compute_confounds = 'compcor_10'

gm_wm_csf_mask = pd.read_csv(os.path.join(dir_fmri_list, 'adnidod_wm_csf_mask_path_list.csv')) 
df_fmri_conf_demog = pd.merge(df_fmri_conf_list, df_demog, on='ID_scan')
df_fmri_conf_demog = pd.merge(df_fmri_conf_demog, gm_wm_csf_mask, on=['ID_scan','ID_subject'])
#df_fmri_conf_demog = pd.concat([df_fmri_conf_list, df_demog, gm_wm_csf_mask], on='ID_scan', ignore_index=True)
# df_fmri_mask

func_imgs   = list(df_fmri_conf_demog['fmri_path'])
#gm_mask     = [df_fmri_conf_demog['mask_wm'], df_fmri_conf_demog['mask_csf']]
gm_mask     = list(df_fmri_conf_demog['mask_gm'])
groups      = list(df_fmri_conf_demog['ID_subject'])
phenotypic  = df_fmri_conf_demog

# condounds
# confounds = {None, path}
# compute_confounds = [None, 'compcor_5', 'compcor_10'}
confounds           = list(df_fmri_conf_demog['conf_path'])
compute_confounds   = None

#phenotypic = df_fmri_conf_demog.loc[:, ['age', 'hand', 'marry',
#       'educat', 'retired', 'ethnic', 'racial']]

#abide_data = datasets.fetch_abide_pcp(pipeline='cpac')
#func_imgs = abide_data.func_preproc
#phenotypic = abide_data.phenotypic

# class type for each subject is different
class_type = 'diagnosis' # name of the column in the df 
cache_path = '/volatile/darya/Documents/experiments/analysis/cache'

from sklearn.externals.joblib import Memory, Parallel, delayed
mem = Memory(cachedir=cache_path)

connectome_regress_confounds = None

from nilearn_utils import data_info
target_shape, target_affine, _ = data_info(func_imgs[0])


############################################################################
## Data
## ----
## Load the datasets from Nilearn
#
#from nilearn import datasets
#
#abide_data = datasets.fetch_abide_pcp(pipeline='cpac')
#func_imgs = abide_data.func_preproc
#phenotypic = abide_data.phenotypic
#
## class type for each subject is different
#class_type = 'DX_GROUP'
#cache_path = 'data_processing_abide'
#
#from sklearn.externals.joblib import Memory, Parallel, delayed
#mem = Memory(cachedir=cache_path)
#
#connectome_regress_confounds = None
#
#from nilearn_utils import data_info
#target_shape, target_affine, _ = data_info(func_imgs[0])

###########################################################################
# Masker
# ------
# Masking the data

# Fetch grey matter mask from nilearn shipped with ICBM templates
from nilearn import datasets
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=gm_mask, target_shape=target_shape,
                          target_affine=target_affine, smoothing_fwhm=6.,
                          standardize=True, detrend=True, mask_strategy='epi',
                          memory=mem, memory_level=2, n_jobs=10,
                          verbose=5)

##############################################################################
# Cross Validator
# ---------------

#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

#classes = df_fmri_conf_demog[class_type].values # 0 - NC, 1 - PTSD
classes = phenotypic[class_type].values

_, labels = np.unique(classes, return_inverse=True)
#cv = StratifiedShuffleSplit(labels, n_iter=20, test_size=0.25, random_state=0)
cv = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=0)

###############################################################################
# Atlases
# ---------------------------------------
from nilearn import datasets
from nilearn import plotting

atas_yeo = datasets.fetch_atlas_yeo_2011()
atlas_aal = datasets.fetch_atlas_aal()
atlas_basc_multiscale = datasets.fetch_atlas_basc_multiscale_2015(version='asym') 
atlas_basc_multiscale_sym = datasets.fetch_atlas_basc_multiscale_2015(version='sym') 
#atlas_craddock = datasets.fetch_atlas_craddock_2012()
atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')   
#atlas_msdl = datasets.fetch_atlas_msdl()                  
#atlas_smith = datasets.fetch_atlas_smith_2009()
atlases = {'yeo': atas_yeo.thick_17,
           'aal': atlas_aal.maps,
           'basc_scale036': atlas_basc_multiscale.scale036,
           'basc_scale122': atlas_basc_multiscale.scale122,
           'basc_scale122_sym': atlas_basc_multiscale_sym.scale122,
           'destrieux': atlas_destrieux.maps,
           'ho': atlas_harvard_oxford.maps}
#for name, atlas in sorted(atlases.items()):
#    plotting.plot_roi(atlas, title=name)


##############################################################################
# Functional Connectivity Analysis model
# ---------------------------------------
from learn import LearnBrainRegions

models = ['kmeans', 'ward', 'ica', 'dictlearn'] #['ica', 'dictlearn']
connectomes = ['correlation', 'partial correlation', 'tangent']

###############################################################################
# Gather results - Data structure

columns = ['iter','atlas', 'measure', 'classifier', 'scores']
gather_results = dict()
for label in columns:
    gather_results.setdefault(label, [])  


from _gather_results import _set_default_dict_with_labels, _append_results

columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'n_regions', 'smoothing_fwhm', 'dataset', 'compcor_10', 'motion_regress',
           'n_components', 'n_clusters', 'connectome_regress', 'scoring']
results = _set_default_dict_with_labels(columns)
print(results)          


##############################################################################
# Run the analysis now
# --------------------

# You can use Parallel if you want here!

for model in models:
    meta_results = Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=None,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=atlases,
            masker=masker, connectomes=connectomes,
            connectome_regress_confounds=connectome_regress_confounds, 
            compute_confounds=compute_confounds,confounds=confounds)
        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
            #for train, test in gss.split(func_imgs, labels, groups=groups):
        #for index, (train_index, test_index) in enumerate(cv))
    for i, meta_result_ in enumerate(meta_results):
        # This needs to be changed according to connectomes and classifiers
        # selected in the analysis.
        gather_results = _append_results(gather_results, meta_result_)

##############################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------
import pandas as pd

results = pd.DataFrame(gather_results)
results.to_csv(os.path.join(cache_path, 'results_conf_atlases_200iter.csv'))

####################################################################

filename = confounds[9]
aa = np.genfromtxt(filename)

fmri = func_imgs[9]

import nibabel as nib
fmri_img = nib.load(fmri)
fmri_img_data = fmri_img.get_data()

aa.shape
fmri_img_data.shape

###

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([1, 1, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)


from sklearn.model_selection import GroupShuffleSplit
cv = GroupShuffleSplit(labels, n_iter=20, test_size=0.25, random_state=0)


gss = GroupShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
for train, test in gss.split(func_imgs, labels, groups=groups):
    print("%s %s" % (train, test))
    
    
for index, (train_index, test_index) in enumerate(gss.split(func_imgs, labels, groups=groups)):
    print("%s %s %s" % (index, train, test))


a = gss.split(func_imgs, labels, groups=groups)

for b,c in a:
    print(b,c)





