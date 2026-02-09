import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Ожидается pandas DataFrame, но получено {type(df)}")
        if df.empty:
            raise ValueError("пустой DataFrame")
        self.df = df.copy()
        self.original_df = df.copy()
        self.transformations = {}
        self.scalers = {}
        self.one_hot_columns = []
        self.removed_columns = []
        self.imputation_values = {}

    def remove_missing(self, threshold=0.5, strategy='median'):
        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold должеен быть в интервале от 0 до 1, получен {threshold}")
        if strategy not in ['mean','median','mode']:
            raise ValueError(f"strategy должен быть 'mean', 'median' или 'mode', получено {strategy}")

        missing_ratio= self.df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if columns_to_drop:
            self.df =self.df.drop(columns=columns_to_drop)
            self.removed_columns.extend(columns_to_drop)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.df.columns:
            if  self.df[col].isnull().any(): #сюд
                self.df[col].isnull().sum()
                if col in numeric_cols:
                    if strategy=='mean':
                        fill_value = self.df[col].mean()
                    elif strategy=='median':
                        fill_value = self.df[col].median()
                    else:
                        fill_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 0
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.imputation_values[col] = fill_value
                elif col in categorical_cols:
                    fill_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.imputation_values[col] = fill_value
        self.transformations['missing_removed'] =True
        self.transformations['missing_threshold']= threshold
        self.transformations['missing_strategy'] = strategy
        return self

    def encode_categorical(self, max_categories=10, drop_first=True):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_cols:
            return self
        cols_to_encode = []
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > max_categories:
                continue
            else:
                cols_to_encode.append(col)
        if not cols_to_encode:
            return self
        for col in cols_to_encode:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first, dtype=int)
            self.df = self.df.drop(columns=[col])
            self.df = pd.concat([self.df, dummies], axis=1)
            new_columns = dummies.columns.tolist()
            self.one_hot_columns.extend(new_columns)

        self.transformations['categorical_encoded'] = True
        self.transformations['max_categories'] = max_categories

        return self

    def normalize_numeric(self, method='minmax', exclude_cols=None):
        if method not in ['minmax', 'std']:
            raise ValueError(f"Метод должен быть 'minmax' или 'std', получен{method}")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        if not numeric_cols:
            return self
        for col in numeric_cols:
            if method =='minmax':
                scaler=MinMaxScaler()
            else:
                scaler = StandardScaler()

            self.scalers[col] = scaler
            col_values = self.df[[col]].values

            try:
                normalized_values = scaler.fit_transform(col_values)
                self.df[col] = normalized_values.flatten()
            except Exception as e:
                pass

        self.transformations['numeric_normalized'] = True
        self.transformations['normalization_method'] = method

        return self

    def fit_transform(self,
                     missing_threshold=0.5,
                     missing_strategy='median',
                     encode_max_categories=10,
                     encode_drop_first=True,
                     normalize_method='minmax',
                     normalize_exclude=None):

        self.df = self.original_df.copy()
        self.scalers = {}
        self.one_hot_columns = []
        self.removed_columns = []
        self.imputation_values = {}
        self.transformations = {}

        try:
            self.remove_missing(threshold=missing_threshold, strategy=missing_strategy)
            self.encode_categorical(max_categories=encode_max_categories, drop_first=encode_drop_first)
            self.normalize_numeric(method=normalize_method, exclude_cols=normalize_exclude)

            return self.df

        except Exception as e:
            raise

    def print_summary(self):
        pass

    def get_transformation_info(self):
        return {
            'transformations': self.transformations,
            'removed_columns': self.removed_columns,
            'one_hot_columns': self.one_hot_columns,
            'scalers': list(self.scalers.keys()),
            'imputation_values': self.imputation_values,
        }

    def transform_new_data(self, new_df):
        if not self.transformations.get('missing_removed', False):
            raise ValueError("Сначала необходимо выполнить fit_transform() на обучающих данных")

        df_transformed = new_df.copy()

        if self.removed_columns:
            columns_to_drop = [col for col in self.removed_columns if col in df_transformed.columns]
            if columns_to_drop:
                df_transformed = df_transformed.drop(columns=columns_to_drop)

        for col, value in self.imputation_values.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(value)

        for col in self.one_hot_columns:
            if col not in df_transformed.columns:
                df_transformed[col] = 0

        for col, scaler in self.scalers.items():
            if col in df_transformed.columns:
                col_values = df_transformed[[col]].values
                normalized_values = scaler.transform(col_values)
                df_transformed[col] = normalized_values.flatten()

        final_columns = list(self.df.columns)
        missing_in_new = set(final_columns) - set(df_transformed.columns)
        for col in missing_in_new:
            df_transformed[col] = 0

        df_transformed = df_transformed[final_columns]
        return df_transformed