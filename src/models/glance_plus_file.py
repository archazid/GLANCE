# -*- coding:utf-8 -*-

import os
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from src.utils.config import USE_CACHE
from src.models.glance import (
    Glance,
)  # We inherit the line-level logic from the original Glance
from src.utils.feature_extractor import extract_hybrid_features


class GlancePlus_File(Glance):
    """
    An enhanced GLANCE model that uses:
    1. A powerful LightGBM classifier for file-level prediction.
    2. A LEXICAL-ONLY feature set (TF-IDF), removing the need for complex feature extraction.
    3. The original, simple line-level ranking heuristic from the Glance base class.
    """

    model_name = "GlancePlus-File"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test=False,
        is_realistic=False,
    ):
        # Initialize the parent Glance class, which also calls BaseModel to load data
        super().__init__(
            train_release, test_release, line_threshold, is_realistic=is_realistic
        )

        # We use TfidfVectorizer for lexical features, which is often better than CountVectorizer
        self.vectorizer = TfidfVectorizer(lowercase=False, min_df=3, max_features=5000)

        # Initialize the LightGBM classifier
        self.clf = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            # Handle class imbalance directly in the model
            is_unbalance=True,
        )

    def file_level_prediction(self):
        """
        This method OVERRIDES the file_level_prediction from the BaseModel.
        """
        print(
            f"Starting file-level prediction for {self.test_release} using {self.model_name}"
        )
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            print("Loading cached file-level results.")
            return

        # 1. Create the lexical feature matrix for the training data
        print("Vectorizing training data...")
        X_train = self.vectorizer.fit_transform(self.train_text)
        y_train = self.train_label
        print(f"Training feature matrix shape: {X_train.shape}")

        # 2. Create the lexical feature matrix for the test data
        print("Vectorizing test data...")
        X_test = self.vectorizer.transform(self.test_text)
        print(f"Test feature matrix shape: {X_test.shape}")

        # 3. Train the LightGBM classifier
        print("Training LightGBM model...")
        self.clf.fit(X_train, y_train)
        print("Training complete.")

        # 4. Predict labels and probabilities for the test set
        self.test_pred_labels = self.clf.predict(X_test)
        # Probability of the positive class
        self.test_pred_scores = self.clf.predict_proba(X_test)[:, 1]

        # 5. Save the results using the method from BaseModel
        self.save_file_level_result()
        print("File-level prediction complete and results saved.")

    # NOTE: We DO NOT re-implement line_level_prediction here.
    # By inheriting from `Glance`, this class will automatically use the original,
    # simple line-level ranking heuristic after its superior file-level prediction is done.


# Note: Will be implemented on Phase 2 of my research plan
class GlancePlus_File_Hybrid(Glance):
    """
    A professional-grade GLANCE model that uses:
    1. A powerful LightGBM classifier for file-level prediction.
    2. A rich, hybrid feature set combining lexical, process, and complexity metrics.
    3. The original, simple line-level ranking heuristic from the Glance base class.
    """

    model_name = "GlancePlus-File-Hybrid"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test=False,
        is_realistic=False,
    ):
        # Initialize the parent Glance class, which also calls BaseModel to load data
        super().__init__(
            train_release, test_release, line_threshold, is_realistic=is_realistic
        )

        # We use TfidfVectorizer for lexical features
        self.vectorizer = TfidfVectorizer(lowercase=False, min_df=3, max_features=5000)

        # Initialize the LightGBM classifier with parameters
        self.clf = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            # Handle class imbalance directly in the model
            is_unbalance=True,
        )

    def _create_hybrid_feature_matrix(
        self, release_name, file_list, text_data, fit_vectorizer=False
    ):
        """
        Helper function to create the final hybrid feature matrix.
        """
        # Extract Process, Complexity, and CK metrics
        traditional_features_df = extract_hybrid_features(release_name, file_list)

        # Extract Lexical features (TF-IDF)
        if fit_vectorizer:
            lexical_features = self.vectorizer.fit_transform(text_data)
        else:
            lexical_features = self.vectorizer.transform(text_data)

        # Align and combine feature sets
        # Ensure the traditional features are in the same order as the text data
        aligned_traditional_features = traditional_features_df.reindex(
            file_list
        ).fillna(0)

        # Convert to a sparse matrix format for efficient combination
        traditional_features_sparse = csr_matrix(aligned_traditional_features.values)

        # Combine sparse matrices horizontally
        hybrid_features = hstack([lexical_features, traditional_features_sparse])

        return hybrid_features

    def file_level_prediction(self):
        """
        This method OVERRIDES the file_level_prediction from the BaseModel.
        It implements the training and prediction logic for our new hybrid classifier.
        """
        print(
            f"Starting file-level prediction for {self.test_release} using {self.model_name}"
        )
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            print("Loading cached file-level results.")
            return

        # Create the hybrid feature matrix for the training data
        # The `fit_vectorizer=True` flag tells the helper to fit the TF-IDF vectorizer
        X_train = self._create_hybrid_feature_matrix(
            self.train_release,
            self.train_filename,
            self.train_text,
            fit_vectorizer=True,
        )
        y_train = self.train_label

        # Create the hybrid feature matrix for the test data
        # Here, we only transform the test data using the already-fitted vectorizer
        X_test = self._create_hybrid_feature_matrix(
            self.test_release, self.test_filename, self.test_text, fit_vectorizer=False
        )

        # Train the LightGBM classifier
        print("Training LightGBM model...")
        self.clf.fit(X_train, y_train)
        print("Training complete.")

        # Predict labels and probabilities for the test set
        self.test_pred_labels = self.clf.predict(X_test)
        self.test_pred_scores = self.clf.predict_proba(X_test)[
            :, 1
        ]  # Probability of the positive class

        # Save the results using the method from BaseModel
        self.save_file_level_result()
        print("File-level prediction complete and results saved.")

    # NOTE: We DO NOT re-implement line_level_prediction here.
    # By inheriting from `Glance`, this class will automatically use the original,
    # simple line-level ranking heuristic after its superior file-level prediction is done.
