"""Processing pipeline example for resting state fMRI datasets
"""
import os
import itertools
import numpy as np



def draw_predictions(imgs=None, labels=None, index=None,
                     train_index=None, test_index=None,
                     dimensionality=None,
                     scoring='roc_auc', models=None, atlases=None,
                     masker=None, connectomes=None,
                     compute_confounds = None,
                     confounds=None, connectome_regress_confounds=None):
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
        n_parcels=dimensionality, #n_parcels = 120,
        n_comp=dimensionality, #n_comp=40,
        #n_clusters=120,
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

###### in case of conf calculation compute_confounds = 'compcor_10'

gm_wm_csf_mask = pd.read_csv(os.path.join(dir_fmri_list, 'adnidod_wm_csf_mask_path_list.csv')) 
df_fmri_conf_demog = pd.merge(df_fmri_conf_list, df_demog, on='ID_scan')
df_fmri_conf_demog = pd.merge(df_fmri_conf_demog, gm_wm_csf_mask, on=['ID_scan','ID_subject'])

func_imgs   = list(df_fmri_conf_demog['fmri_path'][63:73])
#gm_mask     = [df_fmri_conf_demog['mask_wm'], df_fmri_conf_demog['mask_csf']]
gm_mask     = list(df_fmri_conf_demog['mask_gm'][63:73])
groups      = list(df_fmri_conf_demog['ID_subject'][63:73])
phenotypic  = df_fmri_conf_demog[63:73]

# condounds
# confounds = {None, path}
# compute_confounds = [None, 'compcor_5', 'compcor_10'}
confounds           = list(df_fmri_conf_demog['conf_path'][63:73])
compute_confounds   = None

# class type for each subject is different
class_type = 'diagnosis' # name of the column in the df 
cache_path = '/volatile/darya/Documents/experiments/analysis/cache'

from sklearn.externals.joblib import Memory, Parallel, delayed
mem = Memory(cachedir=cache_path)

connectome_regress_confounds = None

from nilearn_utils import data_info
target_shape, target_affine, _ = data_info(func_imgs[0])

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
                          verbose=10)

##############################################################################
# Cross Validator
# ---------------

#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
n_iter = 2

#classes = df_fmri_conf_demog[class_type].values # 0 - NC, 1 - PTSD
classes = phenotypic[class_type].values

_, labels = np.unique(classes, return_inverse=True)
#cv = StratifiedShuffleSplit(labels, n_iter=20, test_size=0.25, random_state=0)
cv = GroupShuffleSplit(n_splits=n_iter, test_size=0.3, random_state=0)

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

models = ['kmeans', 'ward', 'ica', 'dictlearn'] #['ica', 'dictlearn']
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
           'n_components', 'n_clusters', 'connectome_regress', 'confounds', 'compute_confounds', 'scoring']
results = _set_default_dict_with_labels(columns)
print(results)          


##############################################################################
# Run the analysis
# --------------------

dimensions = [40, 60, 80, 100, 120, 150, 200]

import datetime
import nibabel as nib

start = datetime.datetime.now()

folder_name = dataname + '_iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward_dimen'

am = []

for model in models: #['kmeans']:
    print(model)
    am.append(model)
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
            connectome_regress_confounds=connectome_regress_confounds, key=(index, train_index, test_index))
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
            results1 = _append_results(results, all_results_, i, dataname)
results1_csv = pd.DataFrame({k : pd.Series(v) for k, v in results1.iteritems()})

results = _set_default_dict_with_labels(columns)

for key, value in atlases.iteritems():
    print({key: value})
    am.append(key)
        #aa.append(model)
        #print(model)
    all_results = Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=None, atlases={key: value},
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds, key=(index, train_index, test_index))
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
            results2 = _append_results(results, all_results_, i, dataname)
results2_csv = pd.DataFrame({k : pd.Series(v) for k, v in results2.iteritems()})

print('Time of execution ')
print datetime.datetime.now() - start


print(am)
            
################ Save ########################################################

results_csv = pd.concat([results1_csv,results2_csv],axis=0)


#results1_csv = pd.DataFrame({k : pd.Series(v) for k, v in results.iteritems()})

#results2_csv = pd.DataFrame({k : pd.Series(v) for k, v in results.iteritems()})

#results1_csv = pd.DataFrame.from_dict(results)
results_csv.to_csv(os.path.join(cache_path, 'results_conf_atlases_models_100iter_3.csv'))
            
########################################################################   


print(results_csv['atlas'].value_counts())
#results2_csv['atlas'].value_counts()

# for key, value in atlases.iteritems():
#     print({key: value})


#print(results1_csv['atlas'].value_counts())


#
#for model_ in all_results_.models_:
#    print(model_)
#
#
#            
#            
# for k in results.keys():
#     print(k, len(results[k]))
#            
#            
#for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)):
#    print(index, len(train_index), len(test_index))         
#            
            
            
#folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
#for model in ['ica', 'dictlearn']: # models
#    for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)):
#        all_results = draw_predictions(
#            imgs=func_imgs,
#            labels=labels, index=index,
#            train_index=train_index, test_index=test_index,
#            scoring='roc_auc', models=model, atlases=None,
#            masker=masker, connectomes=connectomes,
#            compute_confounds = None,
#            #confounds_mask_img=None, #gm_mask,
#            connectome_regress_confounds=connectome_regress_confounds)
#        print(index)
#        # Dump the results
#        for model_ in all_results.models_:
#            print('model', model_)
#            save_path = os.path.join(cache_path, folder_name, model_, str(index))
#            print(save_path)
#            if not os.path.exists(save_path):
#                print("Making directory {0}".format(save_path))
#                os.makedirs(save_path)
#            # parcellations
#            direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
#            atlas_parcel = all_results.parcellations_[model_]
#            atlas_parcel.to_filename(direc_parcel)
#            print(direc_parcel)
#            # regions of interest
#            direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
#            atlas_roi = all_results.rois_[model_]
#            atlas_roi.to_filename(direc_roi)
#            print(direc_roi)
#        results = _append_results(results, all_results, index)
#
#
#
#
#
#
#cv = GroupShuffleSplit(n_splits=n_iter, test_size=0.30, random_state=0)
#for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)):
#    print (train_index, test_index)
#
#
##===============================================================================
#, index, (train_index, test_index)
#
#
#
#
#import datetime
#start = datetime.datetime.now()
#
#folder_name = dataname + '_iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
#
#for model in models:
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
#    for i, all_results in enumerate(all_results):
#        print('Index is %s.' % i)
#        print('Results is %s.' % all_results)
#
#        # Dump the results
#        for model_ in all_results.models_:
#            save_path = os.path.join(cache_path, folder_name, model_, str(i))
#            print(save_path)
#            if not os.path.exists(save_path):
#                print("Making directory {0}".format(save_path))
#                os.makedirs(save_path)
#                # parcellations
#                direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
#                atlas_parcel = all_results.parcellations_[model_]
#                atlas_parcel.to_filename(direc_parcel)
#                print(direc_parcel)
#                # regions of interest
#                direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
#                atlas_roi = all_results.rois_[model_]
#                atlas_roi.to_filename(direc_roi)
#                print(direc_roi)
#            results = _append_results(results, all_results, i, dataname)   
#
#print('Time of execution ')
#print datetime.datetime.now() - start


#
# aa = []
# for model in models: #['kmeans']:
#     aa.append(model)
#     print(model)
#     all_results = Parallel(n_jobs=20, verbose=2)(
#         delayed(draw_predictions)(
#             imgs=func_imgs,
#             labels=labels, index=index,
#             train_index=train_index, test_index=test_index,
#             scoring='roc_auc', models=models, atlases=atlases,
#             masker=masker, connectomes=connectomes,
#             compute_confounds = None,
#             #confounds_mask_img=None, #gm_mask,
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
#                 # regions of interest
#             direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
#             atlas_roi = all_results_.rois_[model_]
#
#                 #atlas_roi_img = nib.load(atlas_roi)
#                 #atlas_roi_img.to_filename(direc_roi)
#
#                 #atlas_roi.to_filename(direc_roi)
#             print(direc_roi)
#             results = _append_results(results, all_results_, i, dataname)
#
# print('Time of execution ')
# print datetime.datetime.now() - start
