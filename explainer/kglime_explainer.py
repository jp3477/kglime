"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import copy
from functools import partial
import json

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
# from pyDOE2 import lhs
from scipy.stats.distributions import norm

from lime import explanation
from lime.lime_tabular import LimeTabularExplainer
"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


class CustomLimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self, kernel_fn, verbose=False, random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0,
                    fit_intercept=True,
                    random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]],
                        labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            # clf = Ridge(alpha=0.01,
            #             fit_intercept=True,
            #             random_state=self.random_state)
            clf = DecisionTreeRegressor()
            clf.fit(data, labels, sample_weight=weights)

            # coef = clf.coef_
            coef = clf.feature_importances_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate(
                        (indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata -
                                               num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                # weighted_data = coef * data[0]
                # feature_weights = sorted(zip(range(data.shape[1]),
                #                              weighted_data),
                #                          key=lambda x: np.abs(x[1]),
                #                          reverse=True)
                feature_weights = sorted(zip(range(data.shape[1]), coef),
                                         key=lambda x: np.abs(x[1]),
                                         reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = (
                (data - np.average(data, axis=0, weights=weights)) *
                np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights)) *
                               np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights, num_features,
                                          n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        feature_selection = 'highest_weights'
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column, weights,
                                               num_features, feature_selection)

        used_neighborhood_data = neighborhood_data[:, used_features]

        x_train, x_val, y_train, y_val, sample_weights_train, sample_weights_val = train_test_split(
            used_neighborhood_data,
            labels_column,
            weights,
            random_state=self.random_state)

        if model_regressor is None:
            model_regressor = Ridge(alpha=1,
                                    fit_intercept=True,
                                    random_state=self.random_state)

        model_regressor = DecisionTreeRegressor()
        easy_model = model_regressor

        easy_model.fit(x_train, y_train, sample_weight=sample_weights_train)

        r = permutation_importance(easy_model,
                                   x_val,
                                   y_val,
                                   sample_weight=sample_weights_val,
                                   n_repeats=30,
                                   random_state=self.random_state)

        prediction_score = easy_model.score(neighborhood_data[:,
                                                              used_features],
                                            labels_column,
                                            sample_weight=weights)

        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1))

        # if self.verbose:
        #     print('Intercept', easy_model.intercept_)
        #     print(
        #         'Prediction_local',
        #         local_pred,
        #     )
        #     print('Right:', neighborhood_labels[0, label])
        easy_model.intercept_ = 0.0
        return (easy_model.intercept_,
                sorted(zip(used_features, r.importances_mean),
                       key=lambda x: np.abs(x[1]),
                       reverse=True), prediction_score, local_pred)


class Feature:
    def __init__(self, display_name, name):
        self.display_name = display_name
        self.name = name

    def __repr__(self):
        return self.display_name

    def __eq__(self, other):
        return self.display_name == other.display_name

    def __hash__(self):
        return hash(self.display_name)


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""
    def __init__(self,
                 feature_names,
                 feature_values,
                 feature_real_values,
                 scaled_row,
                 categorical_features,
                 discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            feature_real_values: list of the original data row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_real_values = feature_real_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """

        return self._map_exp_ids_with_features(exp)

    def _map_exp_ids(self, exp):
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def _map_exp_ids_with_features(self, exp):
        names = [
            Feature(display_name, self.feature_real_values[i])
            for i, display_name in enumerate(self.exp_feature_names)
        ]
        if self.discretized_feature_names is not None:
            # names = self.discretized_feature_names

            names = [
                Feature(display_name, self.feature_real_values[i]) for i,
                display_name in enumerate(self.discretized_feature_names)
            ]

        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames, self.feature_values, fweights))
            else:
                out_dict = dict(
                    map(
                        lambda x: (x[0], (x[1], x[2], x[3])),
                        zip(self.feature_indexes, fnames, self.feature_values,
                            fweights)))
                out_list = [
                    out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp
                ]
        else:
            out_list = list(
                zip(self.exp_feature_names, self.feature_values, weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(
            out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeVectorizedExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""
    def __init__(self,
                 mode="classification",
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 sample_around_instance=False,
                 random_state=None,
                 feature_neighbor_fns=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """

        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance

        if categorical_features is None:
            categorical_features = []

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.feature_selection = feature_selection
        self.class_names = class_names

        # Experiment with dynamic kernel weights

        def kernel(d):
            return np.concatenate([
                np.array([1.0]), 1 - MinMaxScaler(clip=True).fit_transform(
                    d[1:].reshape(-1, 1)).flatten()
            ])

        kernel_fn = partial(kernel)

        self.base = CustomLimeBase(kernel_fn,
                                   verbose=verbose,
                                   random_state=self.random_state)

        self.feature_neighbor_fns = feature_neighbor_fns

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1, ),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         model_regressor=None):

        data_row_interpretable, data_row_remaining = np.split(data_row,
                                                              2,
                                                              axis=-1)

        def new_predict_fn(xs):
            n = xs.shape[0]
            remaining_cols = np.repeat(data_row_remaining,
                                       n).reshape(xs.shape, order='F')

            xs = np.hstack((xs, remaining_cols))

            return predict_fn(xs)

        return self._explain_instance(data_row_interpretable,
                                      new_predict_fn,
                                      labels=labels,
                                      top_labels=top_labels,
                                      num_features=num_features,
                                      num_samples=num_samples,
                                      model_regressor=model_regressor)

    def _explain_instance(self,
                          data_row,
                          predict_fn,
                          labels=(1, ),
                          top_labels=None,
                          num_features=10,
                          num_samples=5000,
                          model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        data, inverse, all_dists = self.__data_inverse(data_row, num_samples)

        scaled_data = all_dists

        distances = sklearn.metrics.pairwise_distances(all_dists,
                                                       all_dists[0].reshape(
                                                           1, -1),
                                                       metric='l1').ravel()

        yss, full_data_row = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError(
                    "LIME does not currently support "
                    "classifier models without probability "
                    "scores. If this conflicts with your "
                    "use case, please let us know: "
                    "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                # if not np.allclose(yss.sum(axis=1), 1.0):
                #     warnings.warn("""
                #     Prediction probabilties do not sum to 1, and
                #     thus does not constitute a probability space.
                #     Check that you classifier outputs probabilities
                #     (Not log probabilities, or actual class predictions).
                #     """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(
                                     len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError(
                    "Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(
                        yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)
        feature_indexes = None

        data_row_stacked = np.reshape(full_data_row,
                                      (2, len(self.categorical_features)))
        for i in self.categorical_features:
            name = int(data_row_stacked[0][i])
            name = self.categorical_names[name]
            days_elapsed = int(data_row_stacked[1][i])

            feature_names[i] = '%s=%s (t-%s days)' % (feature_names[i], name,
                                                      days_elapsed)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None

        real_values = list(data_row)
        domain_mapper = TableDomainMapper(
            feature_names,
            values,
            real_values,
            scaled_data[0],
            categorical_features=categorical_features,
            discretized_feature_names=discretized_feature_names,
            feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label], ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                 scaled_data,
                 yss,
                 distances,
                 label,
                 num_features,
                 model_regressor=model_regressor,
                 feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j)
                                    for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse(self, data_row, num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """

        num_cols = data_row.shape[0]
        data = np.zeros((num_samples, num_cols))

        first_row = data_row
        data[0] = data_row.copy()
        inverse = data.copy()
        all_dists = np.zeros((num_samples, num_cols))

        for column in self.categorical_features:
            if data_row[column] in [0]:
                inverse_column = np.repeat(data_row[column], num_samples)
            else:
                if column in self.categorical_features and self.feature_neighbor_fns:
                    values, dists, freqs = self.feature_neighbor_fns(
                        data_row[column])

                else:
                    values = self.feature_values[column].copy()
                    freqs = self.feature_frequencies[column].copy()
                    if 0 in values and len(values) > 1:
                        zero_idx = values.index(0)
                        values.remove(0)
                        freqs = np.delete(freqs, zero_idx)
                        freqs = freqs / np.sum(freqs)

                    dists = np.zeros_like(values)

                sample_indices = self.random_state.choice(range(len(values)),
                                                          size=num_samples,
                                                          replace=True,
                                                          p=freqs).astype(int)

                inverse_column = np.array(values)[sample_indices]
                dists = dists[sample_indices]

            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]

            data[:, column] = binary_column
            inverse[:, column] = inverse_column

            if self.feature_neighbor_fns is not None and data_row[
                    column] not in [0]:
                all_dists[1:, column] = dists[1:]
            else:
                all_dists[:, column] = 0.0

        inverse[0] = data_row

        return data, inverse, all_dists


class KGLIMEExplainer(LimeVectorizedExplainer):
    """
    An explainer for keras-style recurrent neural networks, where the
    input shape is (n_samples, n_timesteps, n_features). This class
    just extends the LimeTabularExplainer class and reshapes the training
    data and feature names such that they become something like

    (val1_t1, val1_t2, val1_t3, ..., val2_t1, ..., valn_tn)

    Each of the methods that take data reshape it appropriately,
    so you can pass in the training/testing data exactly as you
    would to the recurrent neural network.

    """
    def __init__(self,
                 n_timesteps,
                 n_features,
                 mode="classification",
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 random_state=None,
                 feature_neighbor_fns=None):
        """
        Args:
            training_data: numpy 3d array with shape
                (n_samples, n_timesteps, n_features)
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        # Reshape X
        self.n_timesteps = n_timesteps
        self.n_features = n_features

        if feature_names is None:
            feature_names = ['feature%d' % i for i in range(n_features)]

        # Update the feature names
        feature_names = [
            '{}_t-{}'.format(n, n_timesteps - (i + 1)) for n in feature_names
            for i in range(n_timesteps)
        ]

        if categorical_features is None:
            categorical_features = []

        multiplied_categorical_features = []

        cat_feat_idx = 0
        for feature in categorical_features:
            for t in range(self.n_timesteps):
                multiplied_categorical_features.append(cat_feat_idx)
                cat_feat_idx += 1

        # Send off the the super class to do its magic.
        super().__init__(mode=mode,
                         feature_names=feature_names,
                         categorical_features=multiplied_categorical_features,
                         categorical_names=categorical_names,
                         verbose=verbose,
                         class_names=class_names,
                         feature_selection=feature_selection,
                         random_state=random_state,
                         feature_neighbor_fns=feature_neighbor_fns)

    def _make_predict_proba(self, func):
        """
        The predict_proba method will expect 3d arrays, but we are reshaping
        them to 2D so that LIME works correctly. This wraps the function
        you give in explain_instance to first reshape the data to have
        the shape the the keras-style network expects.
        """
        def predict_proba(X):
            n_samples = X.shape[0]
            new_shape = (n_samples, self.n_features, self.n_timesteps)
            X = np.transpose(X.reshape(new_shape), axes=(0, 2, 1))
            return func(X), np.transpose(X, axes=(0, 2, 1)).reshape(
                (n_samples, -1))[0]

        return predict_proba

    def explain_instance(self,
                         data_row,
                         classifier_fn,
                         labels=(1, ),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 2d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities. For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Flatten input so that the normal explainer can handle it
        # data_row = data_row.T.reshape(self.n_timesteps * self.n_features)
        data_row = data_row.T.flatten()

        # Wrap the classifier to reshape input
        classifier_fn = self._make_predict_proba(classifier_fn)
        return super().explain_instance(data_row,
                                        classifier_fn,
                                        labels=labels,
                                        top_labels=top_labels,
                                        num_features=num_features,
                                        num_samples=num_samples,
                                        model_regressor=model_regressor)
