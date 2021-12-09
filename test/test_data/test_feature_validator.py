import copy
import functools

import numpy as np

import pandas as pd

import pytest

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator


# Fixtures to be used in this class. By default all elements have 100 datapoints
@pytest.fixture
def input_data_featuretest(request):
    if request.param == 'numpy_categoricalonly_nonan':
        return np.random.randint(10, size=(100, 10))
    elif request.param == 'numpy_numericalonly_nonan':
        return np.random.uniform(10, size=(100, 10))
    elif request.param == 'numpy_mixed_nonan':
        return np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
    elif request.param == 'numpy_string_nonan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            ['a', 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'numpy_categoricalonly_nan':
        array = np.random.randint(10, size=(100, 10)).astype('float')
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_numericalonly_nan':
        array = np.full(fill_value=10.0, shape=(100, 10), dtype=np.float64)
        array[50, 0:5] = np.nan
        # Somehow array is changed to dtype object after np.nan
        return array.astype('float')
    elif request.param == 'numpy_mixed_nan':
        array = np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_string_nan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            [np.nan, 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'pandas_categoricalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nonan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_categoricalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2, 'C': 8},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_string_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='string')
    elif request.param == 'list_categoricalonly_nonan':
        return [
            ['a', 'b', 'c', 'd'],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nonan':
        return [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nonan':
        return [
            ['a', 2, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif request.param == 'list_categoricalonly_nan':
        return [
            ['a', 'b', 'c', np.nan],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nan':
        return [
            [1, 2, 3, np.nan],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nan':
        return [
            ['a', np.nan, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif 'sparse' in request.param:
        # We expect the names to be of the type sparse_csc_nonan
        sparse_, type_, nan_ = request.param.split('_')
        if 'nonan' in nan_:
            data = np.ones(3)
        else:
            data = np.array([1, 2, np.nan])

        # Then the type of sparse
        row_ind = np.array([0, 1, 2])
        col_ind = np.array([1, 2, 1])
        if 'csc' in type_:
            return sparse.csc_matrix((data, (row_ind, col_ind)))
        elif 'csr' in type_:
            return sparse.csr_matrix((data, (row_ind, col_ind)))
        elif 'coo' in type_:
            return sparse.coo_matrix((data, (row_ind, col_ind)))
        elif 'bsr' in type_:
            return sparse.bsr_matrix((data, (row_ind, col_ind)))
        elif 'lil' in type_:
            return sparse.lil_matrix((data))
        elif 'dok' in type_:
            return sparse.dok_matrix(np.vstack((data, data, data)))
        elif 'dia' in type_:
            return sparse.dia_matrix(np.vstack((data, data, data)))
        else:
            ValueError("Unsupported indirect fixture {}".format(request.param))
    elif 'openml' in request.param:
        _, openml_id = request.param.split('_')
        X, y = sklearn.datasets.fetch_openml(data_id=int(openml_id),
                                             return_X_y=True, as_frame=True)
        return X
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Actual checks for the features
@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_numericalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_numericalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_numericalonly_nonan',
        'pandas_mixed_nonan',
        'pandas_numericalonly_nan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
        'sparse_lil_nan',
        'openml_40981',  # Australian
    ),
    indirect=True
)
def test_featurevalidator_supported_types(input_data_featuretest):
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    else:
        assert isinstance(transformed_X, np.ndarray)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_string_nonan',
        'numpy_string_nan',
    ),
    indirect=True
)
def test_featurevalidator_unsupported_numpy(input_data_featuretest):
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r".*When providing a numpy array.*not supported."):
        validator.fit(input_data_featuretest)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nan',
        'pandas_mixed_nan',
        'openml_179',  # adult workclass has NaN in columns
    ),
    indirect=True
)
def test_featurevalidator_categorical_nan(input_data_featuretest):
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert any(pd.isna(input_data_featuretest))
    categories_ = validator.column_transformer.\
        named_transformers_['categorical_pipeline'].named_steps['onehotencoder'].categories_
    assert any(('0' in categories) or (0 in categories) or ('missing_value' in categories) for categories in
               categories_)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted
    assert isinstance(transformed_X, np.ndarray)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
    ),
    indirect=True
)
def test_featurevalidator_fitontypeA_transformtypeB(input_data_featuretest):
    """
    Check if we can fit in a given type (numpy) yet transform
    if the user changes the type (pandas then)

    This is problematic only in the case we create an encoder
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    if isinstance(input_data_featuretest, pd.DataFrame):
        pytest.skip("Column order change in pandas is not supported")
    elif isinstance(input_data_featuretest, np.ndarray):
        complementary_type = validator.numpy_to_pandas(input_data_featuretest)
    elif isinstance(input_data_featuretest, list):
        complementary_type, _ = validator.list_to_pandas(input_data_featuretest)
    elif sparse.issparse(input_data_featuretest):
        complementary_type = sparse.csr_matrix(input_data_featuretest.todense())
    else:
        raise ValueError(type(input_data_featuretest))
    transformed_X = validator.transform(complementary_type)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


def test_featurevalidator_get_columns_to_encode():
    """
    Makes sure that encoded columns are returned by _get_columns_to_encode
    whereas numerical columns are not returned
    """
    validator = TabularFeatureValidator()

    df = pd.DataFrame([
        {'int': 1, 'float': 1.0, 'category': 'one', 'bool': True},
        {'int': 2, 'float': 2.0, 'category': 'two', 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    validator.fit(df)

    categorical_columns, numerical_columns, feat_type = validator._get_columns_info(df)

    assert numerical_columns == ['int', 'float']
    assert categorical_columns == ['category', 'bool']
    assert feat_type == ['numerical', 'numerical', 'categorical', 'categorical']


def feature_validator_remove_nan_catcolumns(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                            ans_train: np.ndarray, ans_test: np.ndarray) -> None:
    validator = TabularFeatureValidator()
    validator.fit(df_train)
    transformed_df_train = validator.transform(df_train)
    transformed_df_test = validator.transform(df_test)

    assert np.array_equal(transformed_df_train, ans_train)
    assert np.array_equal(transformed_df_test, ans_test)


def test_feature_validator_remove_nan_catcolumns():
    """
    Make sure categorical columns that have only nan values are removed.
    Transform performs the folloing:
        * simple imputation for both
        * scaling for numerical
        * one-hot encoding for categorical
    For example,
        data = [
            {'A': 1, 'B': np.nan, 'C': np.nan},
            {'A': np.nan, 'B': 3, 'C': np.nan},
            {'A': 2, 'B': np.nan, 'C': np.nan}
        ]
    and suppose all the columns are categorical,
    then
        * `A` in {np.nan, 1, 2}
        * `B` in {np.nan, 3}
        * `C` in {np.nan} <=== it will be dropped.

    So in the column A,
        * np.nan ==> [1, 0, 0]
        * 1      ==> [0, 1, 0]
        * 2      ==> [0, 0, 1]
    in the column B,
        * np.nan ==> [1, 0]
        * 3      ==> [0, 1]
    Therefore, by concatenating,
        * {'A': 1, 'B': np.nan, 'C': np.nan} ==> [0, 1, 0, 1, 0]
        * {'A': np.nan, 'B': 3, 'C': np.nan} ==> [1, 0, 0, 0, 1]
        * {'A': 2, 'B': np.nan, 'C': np.nan} ==> [0, 0, 1, 1, 0]
    """
    # First case, there exist null columns (B and C) in the train set
    # and a same column (C) are not all null for the test set.

    df_train = pd.DataFrame(
        [
            {'A': 1, 'B': np.nan, 'C': np.nan},
            {'A': np.nan, 'C': np.nan},
            {'A': 1}
        ],
        dtype='category',
    )
    ans_train = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float64)
    df_test = pd.DataFrame(
        [
            {'A': np.nan, 'B': np.nan, 'C': 5},
            {'A': np.nan, 'C': np.nan},
            {'A': 1}
        ],
        dtype='category',
    )
    ans_test = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
    feature_validator_remove_nan_catcolumns(df_train, df_test, ans_train, ans_test)

    # Second case, there exist null columns (B and C) in the training set and
    # the same columns (B and C) are null in the test set.
    df_train = pd.DataFrame(
        [
            {'A': 1, 'B': np.nan, 'C': np.nan},
            {'A': np.nan, 'C': np.nan},
            {'A': 1}
        ],
        dtype='category',
    )
    ans_train = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float64)
    df_test = pd.DataFrame(
        [
            {'A': np.nan, 'B': np.nan, 'C': np.nan},
            {'A': np.nan, 'C': np.nan},
            {'A': 1}
        ],
        dtype='category',
    )
    ans_test = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
    feature_validator_remove_nan_catcolumns(df_train, df_test, ans_train, ans_test)

    # Third case, there exist no null columns in the training set and
    # null columns exist in the test set.
    df_train = pd.DataFrame(
        [
            {'A': 1, 'B': 1},
            {'A': 2, 'B': 2}
        ],
        dtype='category',
    )
    ans_train = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    df_test = pd.DataFrame(
        [
            {'A': np.nan, 'B': np.nan},
            {'A': np.nan, 'B': np.nan}
        ],
        dtype='category',
    )
    ans_test = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    feature_validator_remove_nan_catcolumns(df_train, df_test, ans_train, ans_test)


def test_features_unsupported_calls_are_raised():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input or using the validator in a way that is not
    expected
    """
    validator = TabularFeatureValidator()
    with pytest.raises(TypeError, match=r".*?Convert the time information to a numerical value"):
        validator.fit(
            pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
        )
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r"AutoPyTorch only supports.*yet, the provided input"):
        validator.fit({'input1': 1, 'input2': 2})
    validator = TabularFeatureValidator()
    with pytest.raises(TypeError, match=r".*?but input column A has an invalid type `string`.*"):
        validator.fit(pd.DataFrame([{'A': 1, 'B': 2}], dtype='string'))
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r"The feature dimensionality of the train and test"):
        validator.fit(X_train=np.array([[1, 2, 3], [4, 5, 6]]),
                      X_test=np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
                      )
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r"Cannot call transform on a validator that is not fit"):
        validator.transform(np.array([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
        'sparse_lil_nan',
    ),
    indirect=True
)
def test_no_column_transformer_created(input_data_featuretest):
    """
    Makes sure that for numerical only features, no encoder is created
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    validator.transform(input_data_featuretest)
    assert validator.column_transformer is None


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
    ),
    indirect=True
)
def test_column_transformer_created(input_data_featuretest):
    """
    This test ensures an column transformer is created if categorical data is provided
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert validator.column_transformer is not None

    # Make sure that the encoded features are actually encoded. Categorical columns are at
    # the start after transformation. In our fixtures, this is also honored prior encode
    cat_columns, _, feature_types = validator._get_columns_info(input_data_featuretest)

    # At least one categorical
    assert 'categorical' in validator.feat_type

    # Numerical if the original data has numerical only columns
    if np.any([pd.api.types.is_numeric_dtype(input_data_featuretest[col]
                                             ) for col in input_data_featuretest.columns]):
        assert 'numerical' in validator.feat_type
        # we expect this input to be the fixture 'pandas_mixed_nan'
        np.testing.assert_array_equal(transformed_X, np.array([[1., 0., -1.], [0., 1., 1.]]))
    else:
        np.testing.assert_array_equal(transformed_X, np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]]))

    if not all([feat_type in ['numerical', 'categorical'] for feat_type in feature_types]):
        raise ValueError("Expected only numerical and categorical feature types")


def test_no_new_category_after_fit():
    """
    This test makes sure that we can actually pass new categories to the estimator
    without throwing an error
    """
    # Then make sure we catch categorical extra categories
    x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
    validator = TabularFeatureValidator()
    validator.fit(x)
    x['A'] = x['A'].apply(lambda x: x * x)
    validator.transform(x)


def test_unknown_encode_value():
    x = pd.DataFrame([
        {'a': -41, 'b': -3, 'c': 'a', 'd': -987.2},
        {'a': -21, 'b': -3, 'c': 'a', 'd': -9.2},
        {'a': 0, 'b': -4, 'c': 'b', 'd': -97.2},
        {'a': -51, 'b': -3, 'c': 'a', 'd': 987.2},
        {'a': 500, 'b': -3, 'c': 'a', 'd': -92},
    ])
    x['c'] = x['c'].astype('category')
    validator = TabularFeatureValidator()

    # Make sure that this value is honored
    validator.fit(x)
    x['c'].cat.add_categories(['NA'], inplace=True)
    x.loc[0, 'c'] = 'NA'  # unknown value
    x_t = validator.transform(x)
    # The first row should have a 0, 0 as we added a
    # new categorical there and one hot encoder marks
    # it as all zeros for the transformed column
    expected_row = [0.0, 0.0, -0.5584294383572701, 0.5000000000000004, -1.5136598016833485]
    assert expected_row == x_t[0].tolist()


# Actual checks for the features
@pytest.mark.parametrize(
    'openml_id',
    (
        40981,  # Australian
        3,  # kr-vs-kp
        1468,  # cnae-9
        40975,  # car
        40984,  # Segment
    ),
)
@pytest.mark.parametrize('train_data_type', ('numpy', 'pandas', 'list'))
@pytest.mark.parametrize('test_data_type', ('numpy', 'pandas', 'list'))
def test_feature_validator_new_data_after_fit(
    openml_id,
    train_data_type,
    test_data_type,
):

    # List is currently not supported as infer_objects
    # cast list objects to type objects
    if train_data_type == 'list' or test_data_type == 'list':
        pytest.skip()

    validator = TabularFeatureValidator()

    if train_data_type == 'numpy':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=False)
    elif train_data_type == 'pandas':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
    else:
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
        X = X.values.tolist()
        y = y.values.tolist()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    validator.fit(X_train)

    transformed_X = validator.transform(X_test)

    # Basic Checking
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    else:
        assert isinstance(transformed_X, np.ndarray)

    # And then check proper error messages
    if train_data_type == 'pandas':
        old_dtypes = copy.deepcopy(validator.dtypes)
        validator.dtypes = ['dummy' for dtype in X_train.dtypes]
        with pytest.raises(ValueError,
                           match=r"The dtype of the features must not be changed after fit"):
            transformed_X = validator.transform(X_test)
        validator.dtypes = old_dtypes
        if test_data_type == 'pandas':
            columns = X_test.columns.tolist()
            X_test = X_test[reversed(columns)]
            with pytest.raises(ValueError,
                               match=r"The column order of the features must not be changed after fit"):
                transformed_X = validator.transform(X_test)


def test_comparator():
    numerical = 'numerical'
    categorical = 'categorical'

    validator = TabularFeatureValidator

    with pytest.raises(ValueError, match=r"The comparator for the column order only accepts .*"):
        dummy = 'dummy'
        feat_type = [numerical, categorical, dummy]
        feat_type = sorted(
            feat_type,
            key=functools.cmp_to_key(validator._comparator)
        )

    feat_type = [numerical, categorical] * 10
    ans = [categorical] * 10 + [numerical] * 10
    feat_type = sorted(
        feat_type,
        key=functools.cmp_to_key(validator._comparator)
    )
    assert ans == feat_type

    feat_type = [numerical] * 10 + [categorical] * 10
    ans = [categorical] * 10 + [numerical] * 10
    feat_type = sorted(
        feat_type,
        key=functools.cmp_to_key(validator._comparator)
    )
    assert ans == feat_type


def test_feature_validator_imbalanced_data():

    # Null columns in the train split but not necessarily in the test split
    train_features = {
        'A': [np.NaN, np.NaN, np.NaN],
        'B': [1, 2, 3],
        'C': [np.NaN, np.NaN, np.NaN],
        'D': [np.NaN, np.NaN, np.NaN],
    }
    test_features = {
        'A': [3, 4, 5],
        'B': [6, 5, 7],
        'C': [np.NaN, np.NaN, np.NaN],
        'D': ['Blue', np.NaN, np.NaN],
    }

    X_train = pd.DataFrame.from_dict(train_features)
    X_test = pd.DataFrame.from_dict(test_features)
    validator = TabularFeatureValidator()
    validator.fit(X_train)

    train_feature_types = copy.deepcopy(validator.feat_type)
    assert train_feature_types == ['numerical']
    # validator will throw an error if the column types are not the same
    transformed_X_test = validator.transform(X_test)
    transformed_X_test = pd.DataFrame(transformed_X_test)
    assert sorted(validator.all_nan_columns) == sorted(['A', 'C', 'D'])
    # as there are no categorical columns, we can make such an
    # assertion. We only expect to drop the all nan columns
    total_all_nan_columns = len(validator.all_nan_columns)
    total_columns = len(validator.column_order)
    assert total_columns - total_all_nan_columns == len(transformed_X_test.columns)

    # Columns with not all null values in the train split and
    # completely null on the test split.
    train_features = {
        'A': [np.NaN, np.NaN, 4],
        'B': [1, 2, 3],
        'C': ['Blue', np.NaN, np.NaN],
    }
    test_features = {
        'A': [np.NaN, np.NaN, np.NaN],
        'B': [6, 5, 7],
        'C': [np.NaN, np.NaN, np.NaN],
    }

    X_train = pd.DataFrame.from_dict(train_features)
    X_test = pd.DataFrame.from_dict(test_features)
    validator = TabularFeatureValidator()
    validator.fit(X_train)

    train_feature_types = copy.deepcopy(validator.feat_type)
    assert train_feature_types == ['categorical', 'numerical', 'numerical']

    transformed_X_test = validator.transform(X_test)
    transformed_X_test = pd.DataFrame(transformed_X_test)
    assert not len(validator.all_nan_columns)
