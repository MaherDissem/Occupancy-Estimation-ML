from sklearn.decomposition import PCA


def perform_pca(data, n_components):
    """
    Perform Principal Component Analysis (PCA) on the given data.

    Args:
        data (array-like): The input data to perform PCA on.
        n_components (int): The number of components to keep.

    Returns:
        array-like: Transformed data after PCA.
    """
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)


# TODO save plots