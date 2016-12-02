# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:55:22 2016

@author: darya
"""

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


folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'

with Parallel(n_jobs=20) as parallel:
    for model in models:
        all_results= Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds)    
        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
            
            
            
            
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

#=============================================================================

folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
for model in models:
    all_results = Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds)
    for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups))):
        
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
        print(index)
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
            results = _append_results(results, all_results, index)   
        


#=============================================================================

    
#=============================================================================
if __name__ == '__main__': 
    with Parallel(n_jobs=20) as parallel:
        folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
        for model in models: #['ica', 'dictlearn']:
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
                    #key=(train_index, test_index))
                print(index)
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
                results = _append_results(results, all_results, index)


#=============================================================================




folder_name = name + ' iter' + str(n_iter) + '_smooth6_ica_dictlearn_kmeans_ward'
for model in models: #['ica', 'dictlearn']:
    all_results = Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
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
                print(direc_roi))
        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups)))
            
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
        
        
        
        
with Parallel(n_jobs=20) as parallel:
    
    
    
    
    
    
for model in models:
        all_results= Parallel(n_jobs=20, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            compute_confounds = None,
            #confounds_mask_img=None, #gm_mask,
            connectome_regress_confounds=connectome_regress_confounds)    
        for index, (train_index, test_index) in enumerate(cv.split(func_imgs, labels, groups=groups))):




from joblib import Parallel, delayed
def multiple(a, b):
    return a*b

a = np.array([1,2,3,4,5,6])
b = np.array([1,2,3,4,5,6])

c =  Parallel(n_jobs=2)(delayed(multiple)(a=i, b=j) for i in range(1, 6) for j in range(1,6)) 

