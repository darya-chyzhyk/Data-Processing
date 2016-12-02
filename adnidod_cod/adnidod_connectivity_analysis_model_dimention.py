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
                     compute_confounds = None, confounds=None, n_parcels=None,
                     n_comp=None):
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
        n_comp=n_comp,
        n_parcels = n_parcels, #n_clusters=120,
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

    return learn_brain_regions #, key

###########################################################################
# Data
# ----
# ADNIDOD data
import pandas as pd

dataname = 'adnidod'

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
                          memory=mem, memory_level=2, n_jobs=20,
                          verbose=10)

##############################################################################
# Cross Validator
# ---------------

#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
n_iter = 100

#classes = df_fmri_conf_demog[class_type].values # 0 - NC, 1 - PTSD
classes = phenotypic[class_type].values

_, labels = np.unique(classes, return_inverse=True)
#cv = StratifiedShuffleSplit(labels, n_iter=20, test_size=0.25, random_state=0)
cv = GroupShuffleSplit(n_splits=n_iter, test_size=0.25, random_state=0)

###############################################################################
# Atlases
# ---------------------------------------
from nilearn import datasets
from nilearn import plotting

atas_yeo = datasets.fetch_atlas_yeo_2011()
atlas_aal = datasets.fetch_atlas_aal()
atlas_basc_multiscale = datasets.fetch_atlas_basc_multiscale_2015(version='asym') 
atlas_basc_multiscale_sym = datasets.fetch_atlas_basc_multiscale_2015(version='sym') 
atlas_craddock = datasets.fetch_atlas_craddock_2012()
atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
#atlas_msdl = datasets.fetch_atlas_msdl()
atlas_smith = datasets.fetch_atlas_smith_2009()
atlases = {'yeo': atas_yeo.thick_17,
           'aal_spm12': atlas_aal.maps,
           'basc_scale036': atlas_basc_multiscale.scale036,
           'basc_scale122': atlas_basc_multiscale.scale122,
           'basc_scale122_sym': atlas_basc_multiscale_sym.scale122,
           'destrieux': atlas_destrieux.maps,
           'ho_cort_symm_split': atlas_harvard_oxford.maps,
           'craddok': atlas_craddock.random,
           'smith_bm70': atlas_smith.bm70}
#for name, atlas in sorted(atlases.items()):
#    plotting.plot_roi(atlas, title=name)


##############################################################################
# Functional Connectivity Analysis model
# ---------------------------------------
from learn import LearnBrainRegions

models = ['dictlearn'] #['kmeans', 'ward', 'ica', 'dictlearn']
connectomes = ['correlation', 'partial correlation', 'tangent']

###############################################################################
# Gather results - Data structure

#columns = ['iter','atlas', 'measure', 'classifier', 'scores']
#gather_results = dict()
#for label in columns:
#    gather_results.setdefault(label, [])  


from _gather_results import _set_default_dict_with_labels, _append_results

columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'n_regions', 'smoothing_fwhm', 'dataset', 'compcor_10', 'motion_regress',
           'dimensionality', 'connectome_regress', 'confounds', 'compute_confounds', 'scoring']
results = _set_default_dict_with_labels(columns)
print(results)          


##############################################################################
# Run the analysis now
# --------------------

import datetime
import nibabel as nib

dimensions = [80, 100, 120, 150, 200, 300]
start = datetime.datetime.now()

folder_name = dataname + '_confounds_models_dimension_' + str(n_iter) + 'iter'

for model in models:
    for dim in dimensions:
        results = _set_default_dict_with_labels(columns)
        print(model)

            #aa.append(model)
            #print(model)
        all_results = Parallel(n_jobs=20, verbose=2)(
            delayed(draw_predictions)(
                imgs=func_imgs,
                labels=labels, index=index,
                train_index=train_index, test_index=test_index,
                scoring='roc_auc', models=model, atlases=None,
                masker=masker, connectomes=connectomes,
                compute_confounds = None,
                #confounds_mask_img=None, #gm_mask,
                connectome_regress_confounds=connectome_regress_confounds,
                n_parcels=dim, n_comp=dim)
            for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))

        for i, all_results_ in enumerate(all_results):
            print('Index is %s.' % i)
            print('Results is %s.' % all_results)

            # Dump the results
            for model_ in all_results_.models_:
                save_path = os.path.join(cache_path, folder_name, model_, str(i))
                print(save_path)
                if not os.path.exists(save_path):
                    print("Making directory {0}".format(save_path))
                    os.makedirs(save_path)
                        # parcellations
                direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
                atlas_parcel = all_results_.parcellations_[model_]
                if type(atlas_parcel) == str:
                    atlas_parcel_img = nib.load(atlas_parcel)
                    atlas_parcel_img.to_filename(direc_parcel)
                elif type(atlas_parcel) == nib.nifti1.Nifti1Image:
                    atlas_parcel.to_filename(direc_parcel)
                print(direc_parcel)
                        # regions of interest
                direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
                atlas_roi = all_results_.rois_[model_]

                        #atlas_roi_img = nib.load(atlas_roi)
                        #atlas_roi_img.to_filename(direc_roi)

                        #atlas_roi.to_filename(direc_roi)
                print(direc_roi)
                results1 = _append_results(results, all_results_, i, dataname, dim)

        results1_csv = pd.DataFrame(
                {k: pd.Series(v) for k, v in results.iteritems()})
        cvs_name = (
                folder_name + '_' + str(model) + '_' + str(dim) + 'dim.csv')
        results1_csv.to_csv(os.path.join(cache_path, cvs_name))




print('Time of execution ')
print datetime.datetime.now() - start






##############################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------
# import pandas as pd
#
# results_csv = pd.concat([results_models,results_atlases],axis=0)
#
# #results_csv = pd.DataFrame(results)
# results_csv.to_csv(os.path.join(cache_path, 'results_conf_models_atlases_100iter.csv'))

####################################################################




#for model in models: #['kmeans']:
#    print(model)
#    all_results = Parallel(n_jobs=20, verbose=2)(
#        delayed(draw_predictions)(
#            imgs=func_imgs,
#            labels=labels, index=index,
#            train_index=train_index, test_index=test_index,
#            scoring='roc_auc', models=model, atlases=atlases,
#            masker=masker, connectomes=connectomes,
#            compute_confounds = None,
#            #confounds_mask_img=None, #gm_mask,
#            connectome_regress_confounds=connectome_regress_confounds, key=(index, train_index, test_index))
#        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
#        
#    for i, all_results_ in enumerate(all_results):
#        print('Index is %s.' % i)
#        print('Results is %s.' % all_results)
#
#        # Dump the results
#        for model_ in all_results_.models_:
#            save_path = os.path.join(cache_path, folder_name, model_, str(i))
#            print(save_path)
#            if not os.path.exists(save_path):
#                print("Making directory {0}".format(save_path))
#                os.makedirs(save_path)
#                # parcellations
#                direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
#                atlas_parcel = all_results_.parcellations_[model_]
#                atlas_parcel_img = nib.load(atlas_parcel)
#                atlas_parcel_img.to_filename(direc_parcel)
#                #atlas_parcel.to_filename(direc_parcel)
#                print(direc_parcel)
#                # regions of interest
#                direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
#                atlas_roi = all_results_.rois_[model_]
#                
#                atlas_roi_img = nib.load(atlas_roi)
#                atlas_roi_img.to_filename(direc_roi)
#                
#                #atlas_roi.to_filename(direc_roi)
#                print(direc_roi)
#            results = _append_results(results, all_results_, i, dataname)   
#
#
#
#print('Time of execution ')
#print datetime.datetime.now() - start



#
#
# for model in models:  # ['kmeans']:
#     print(model)
#     all_results = Parallel(n_jobs=20, verbose=2)(
#         delayed(draw_predictions)(
#             imgs=func_imgs,
#             labels=labels, index=index,
#             train_index=train_index, test_index=test_index,
#             scoring='roc_auc', models=model, atlases=atlases,
#             masker=masker, connectomes=connectomes,
#             compute_confounds=None,
#             # confounds_mask_img=None, #gm_mask,
#             connectome_regress_confounds=connectome_regress_confounds, key=(index, train_index, test_index))
#         for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
#
#     for i, all_results_ in enumerate(all_results):
#         print('Index is %s.' % i)
#         print('Results is %s.' % all_results)
#
#         # Dump the results
#         for model_ in all_results_.models_:
#             save_path = os.path.join(cache_path, folder_name, model_, str(i))
#             print(save_path)
#             if not os.path.exists(save_path):
#                 print("Making directory {0}".format(save_path))
#                 os.makedirs(save_path)
#                 # parcellations
#             direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
#             atlas_parcel = all_results_.parcellations_[model_]
#             if type(atlas_parcel) == str:
#                 atlas_parcel_img = nib.load(atlas_parcel)
#                 atlas_parcel_img.to_filename(direc_parcel)
#             elif type(atlas_parcel) == nib.nifti1.Nifti1Image:
#                 atlas_parcel.to_filename(direc_parcel)
#             print(direc_parcel)
#             # regions of interest
#             direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
#             atlas_roi = all_results_.rois_[model_]
#
#             # atlas_roi_img = nib.load(atlas_roi)
#             # atlas_roi_img.to_filename(direc_roi)
#
#             # atlas_roi.to_filename(direc_roi)
#             print(direc_roi)
#             results = _append_results(results, all_results_, i, dataname)
#
#










