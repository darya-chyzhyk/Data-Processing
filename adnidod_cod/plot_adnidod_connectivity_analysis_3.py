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
                     compute_confounds = None, confounds=None, key=None): 
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
                          memory=mem, memory_level=2, n_jobs=10,
                          verbose=5)

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
#atlas_craddock = datasets.fetch_atlas_craddock_2012()
atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')   
#atlas_msdl = datasets.fetch_atlas_msdl()                  
#atlas_smith = datasets.fetch_atlas_smith_2009()
atlases = {'yeo': atas_yeo.thick_17,
           'aal_spm12': atlas_aal.maps,
           'basc_scale036': atlas_basc_multiscale.scale036,
           'basc_scale122': atlas_basc_multiscale.scale122,
           'basc_scale122_sym': atlas_basc_multiscale_sym.scale122,
           'destrieux': atlas_destrieux.maps,
           'ho_cort_symm_split': atlas_harvard_oxford.maps}
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
# Run the analysis now
# --------------------

import datetime
import nibabel as nib

start = datetime.datetime.now()

folder_name = dataname + '_iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward_atlases'

for model in models: #['kmeans']:
    print(model)
    all_results = Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=atlases,
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds, key=(index, train_index, test_index))
        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
        
    for i, all_results_ in enumerate(all_results):
        print('Index is %s.' % i)
        print('Results is %s.' % all_results)

        # Dump the results
        for model_ in all_results.models_:
            save_path = os.path.join(cache_path, folder_name, model_, str(i))
            print(save_path)
            if not os.path.exists(save_path):
                print("Making directory {0}".format(save_path))
                os.makedirs(save_path)
                # parcellations
                direc_parcel = os.path.join(save_path, ('parcel_' + str(model_) + '.nii.gz'))
                atlas_parcel = all_results.parcellations_[model_]
                #atlas_parcel_img = nib.load(atlas_parcel)
                #atlas_parcel_img.to_filename(direc_parcel)
                atlas_parcel.to_filename(direc_parcel)
                print(direc_parcel)
                # regions of interest
                direc_roi = os.path.join(save_path, ('roi_' + str(model_) + '.nii.gz'))
                atlas_roi = all_results.rois_[model_]
                atlas_roi.to_filename(direc_roi)
                print(direc_roi)
            results = _append_results(results, all_results, i, dataname)   



print('Time of execution ')
print datetime.datetime.now() - start





##############################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------
import pandas as pd

results_csv = pd.DataFrame(results)
results_csv.to_csv(os.path.join(cache_path, 'results_conf_models_atlases_100iter.csv'))

####################################################################









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



folder_name = dataname + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'

for model in models:
    all_results, index, (train_index, test_index) = Parallel(n_jobs=20, verbose=2)(
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
        
    
    for i, all_results in enumerate(all_results):
        print('Index is %s.' % i)
        print('Index is %s.' % all_results)
        
        
        
        print('Index is %s.' % index)
        all_results = draw_predictions(
        imgs=func_imgs,
        labels=labels, index=index,
        train_index=train_index, test_index=test_index,
        scoring='roc_auc', models=model, atlases=None,
        masker=masker, connectomes=connectomes,
        compute_confounds = None,
        #confounds_mask_img=None, #gm_mask,
        connectome_regress_confounds=connectome_regress_confounds)
        #key=(train_index, test_index))
        print('index')
        # Dump the results
        for model_ in all_results.models_:
            save_path = os.path.join(cache_path, folder_name, model_, str(index))
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
            results = _append_results(results, all_results, index, dataname)   
            
            
            
            

folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
for model in ['ica', 'dictlearn']: # models
    for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)):
        all_results = draw_predictions(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds)
        print(index)
        # Dump the results
        for model_ in all_results.models_:
            print('model', model_)
            save_path = os.path.join(cache_path, folder_name, model_, str(index))
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



#############################################################

def _path_func(data_list, key=None, **path_params):
    ....
    return dict_mf, history, key


if __name__ ==  "__main__":
        # Use joblib.Parallel like a list comprehension
        # pass the wrapped function a key param to be returned
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._cache(_path_func))(
                data_list, alpha=alpha, gamma=gamma, callback=callback,
                radius=radius, key=(a, g, r), **path_params)
            for a, alpha in enumerate(self.alphas_)
            for g, gamma in enumerate(self.gammas_)
            for r, radius in enumerate(self.radii_))

       # process results from the parallel call, binding the keys to corresponding result values
        self.components_ = None
        for _, history, (a, g, r) in results:
            for time, components in enumerate(history):
                if self.components_ is None:
                    self.components_ = np.ndarray(
                        [len(self.alphas_), len(self.gammas_),
                         len(self.radii_),
                         len(history)] + list(components.shape),
                        dtype=data_list[0].dtype)

                # Post processing normalization. Flip signs in each composant
                # positive part is l0 larger than negative part
                for component in components:
                    if np.sum(component > 0) < np.sum(component < 0):
                        component *= -1
                if self.scale_components:
                    S = np.sqrt(np.sum(components ** 2, axis=1))
                    S[S == 0] = 1
                    components /= S[:, np.newaxis]

                self.components_[a, g, r, time] = components


folder_name = dataname + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
for model in models:
    all_results, index, (train_index, test_index) = Parallel(n_jobs=20, verbose=2)(
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
        
    
    for index in all_results:
        print('Index is %s.' % index)
        all_results = draw_predictions(
        imgs=func_imgs,
        labels=labels, index=index,
        train_index=train_index, test_index=test_index,
        scoring='roc_auc', models=model, atlases=None,
        masker=masker, connectomes=connectomes,
        compute_confounds = None,
        #confounds_mask_img=None, #gm_mask,
        connectome_regress_confounds=connectome_regress_confounds)
        #key=(train_index, test_index))
        print('index')
        # Dump the results
        for model_ in all_results.models_:
            save_path = os.path.join(cache_path, folder_name, model_, str(index))
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
            results = _append_results(results, all_results, index, dataname)   
        






###################################################################3333
# You can use Parallel if you want here!

#for model in models:
#    meta_results = Parallel(n_jobs=20, verbose=2)(
#        delayed(draw_predictions)(
#            imgs=func_imgs,
#            labels=labels, index=None,
#            train_index=train_index, test_index=test_index,
#            scoring='roc_auc', models=model, atlases=atlases,
#            masker=masker, connectomes=connectomes,
#            connectome_regress_confounds=connectome_regress_confounds, 
#            compute_confounds=compute_confounds,confounds=confounds)
#        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
#            #for train, test in gss.split(func_imgs, labels, groups=groups):
#        #for index, (train_index, test_index) in enumerate(cv))
#    for i, meta_result_ in enumerate(meta_results):
#        # This needs to be changed according to connectomes and classifiers
#        # selected in the analysis.
#        gather_results = _append_results(gather_results, meta_result_)



folder_name = dataname + '_iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'

for model in models:
    print(model)
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
        
    for i, all_results in enumerate(all_results):
        print('Index is %s.' % i)
        print('Results is %s.' % all_results)

        # Dump the results
        for model_ in all_results.models_:
            save_path = os.path.join(cache_path, folder_name, model_, str(i))
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
            results = _append_results(results, all_results, i, dataname)   

