import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.fixes import _astype_copy_false
from sklearn.feature_extraction.text import _document_frequency, TfidfTransformer
import scipy.sparse as sp
from sklearn.utils import _IS_32BIT
from sklearn.utils.validation import FLOAT_DTYPES


class CustomTfidfTransformer(TfidfTransformer):
    """
    This class is identical to `sklearn.feature_extraction.text.TfidfTransformer`
    except this class uses the formula idf(t) = log(N / df(t)) if `use_idf` is set to `True`.
    More on `sklearn.feature_extraction.text.TfidfTransformer` at
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html.
    """

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        y : None
            This parameter is not needed to compute tf-idf.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # large sparse data is not supported for 32bit platforms because
        # _document_frequency uses np.bincount which works on arrays of
        # dtype NPY_INTP which is int32 for 32bit platforms. See #20923
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, **_astype_copy_false(df))

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # change idf formula
            idf = np.log(n_samples / df)
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self


def make_violin_plot(dataset: pd.core.groupby.generic.SeriesGroupBy, title: str):
    dataset = dataset.apply(list).to_dict()
    labels = list(dataset.keys())
    dataset = [dataset[genre] for genre in labels]
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticklabels([''] + labels)
    ax.grid(visible=True)
    ax.violinplot(dataset, showmeans=True)