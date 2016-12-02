# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:59:37 2016

@author: darya
"""
compute_confounds = None
def transform(self, imgs, confounds=None):
        """Signal extraction from regions learned on the images.

        Parameters
        ----------
        imgs : Nifti like images, list

        confounds : csv file or array-like, optional
            Contains signals like motion, high variance confounds from
            white matter, csf. This is passed to signal.clean
        """
        self._check_fitted()
        SUBJECTS_TIMESERIES = dict()
        models = []
        for model in self.model:
            models.append(model)

        # Getting Masker to transform fMRI images in Nifti to timeseries signals
        # based on atlas learning
        if self.masker is None:
            raise ValueError("Could not find masker attribute. Masker is missing")

        if not isinstance(imgs, collections.Iterable) or \
                isinstance(imgs, _basestring):
            imgs = [imgs, ]

        mask_img = self.masker.mask_img

        if compute_confounds not in ['compcor_5', 'compcor_10', None]:
            warnings.warn("Given invalid input compute_confounds={0}. Given "
                          "input is diverting to compute_confounds=None"
                          .format(compute_confounds))
            compute_confounds = None

        if self.compute_not_mask_confounds not in ['compcor_5', 'compcor_10', None]:
            warnings.warn("Invalid input type of 'compute_not_mask_confounds'={0}"
                          "is provided. Switching to None"
                          .format(self.compute_not_mask_confounds))
            self.compute_not_mask_confounds = None

        if confounds is None and compute_confounds is None and \
                compute_not_mask_confounds is None:
            confounds = [None] * len(imgs)
            print('a')

        if compute_confounds is not None:
            print('a')
            if compute_confounds == 'compcor_5':
                n_confounds = 5
            elif compute_confounds == 'compcor_10':
                n_confounds = 10

            confounds_ = self.masker.memory.cache(compute_confounds)(
                imgs, mask_img, n_confounds=n_confounds)
                
        if compute_confounds is None:
            print('a')
            confounds_ = None
        

        if confounds_ is not None:
            if confounds is not None:
                confounds = np.hstack((confounds, confounds_))
            else:
                confounds = confounds_

        if confounds is not None and isinstance(confounds, collections.Iterable):
            if len(confounds) != len(imgs):
                raise ValueError("Number of confounds given doesnot match with "
                                 "the given number of subjects. Add missing "
                                 "confound in a list.")

        if self.atlases is not None and not \
                isinstance(self.atlases, dict):
            raise ValueError("If 'atlases' are provided, it should be given as "
                             "a dict. Example, atlases={'name': your atlas image}")

        if self.atlases is not None and \
                isinstance(self.atlases, dict):
            for key in self.atlases.keys():
                if self.verbose > 0:
                    print("Found Predefined atlases of name:{0}. Added to "
                          "set of models".format(key))
                self.parcellations_[key] = self.atlases[key]
                self.rois_[key] = self.atlases[key]
                models.append(key)

        self.models_ = models

        for model in self.models_:
            subjects_timeseries = []
            if self.verbose > 0:
                print("[Timeseries Extraction] {0} atlas image is selected"
                      .format(model))
            atlas_img = self.rois_[model]
            masker = check_embedded_atlas_masker(self.masker, atlas_type='auto',
                                                 img=atlas_img, t_r=2.53,
                                                 low_pass=0.1, high_pass=0.01)

            for img, confound in izip(imgs, confounds):
                if self.verbose > 0:
                    print("Confound found:{0} for subject:{1}".format(confound,
                                                                      img))

                signals = masker.fit_transform(img, confounds=confound)
                subjects_timeseries.append(signals)

            if subjects_timeseries is not None:
                SUBJECTS_TIMESERIES[model] = subjects_timeseries
            else:
                warnings.warn("Timeseries signals extraction are found empty "
                              "for model:{0}".format(model))

        self.subjects_timeseries_ = SUBJECTS_TIMESERIES

        if self.connectome_convert:
            if self.connectome_measure is not None:
                if isinstance(self.connectome_measure, collections.Iterable):
                    catalog = self.connectome_measure
                else:
                    if isinstance(self.connectome_measure, _basestring):
                        catalog = [self.connectome_measure, ]
            else:
                warnings.warn("Given connectome_convert=True but connectome "
                              "are given as None. Taking connectome measure "
                              "kind='correlation'", stacklevel=2)
                catalog = ['correlation']

            self.connectome_measures_ = catalog

            connectivities = self._connectome_converter(
                catalog=self.connectome_measures_,
                confounds=self.connectome_confounds)

        return self
        
        
        
im1 = func_imgs[0]
conf1 = confounds[0]

a = transform(im1, confounds=conf1)