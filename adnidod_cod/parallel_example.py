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
