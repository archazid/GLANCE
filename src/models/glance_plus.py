# -*- coding:utf-8 -*-
import os
import lightgbm as lgb
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from src.utils.config import USE_CACHE
from src.models.base_model import BaseModel  # Inherit from the base class
from src.utils.feature_extractor import (
    extract_hybrid_features,
)  # For the enhanced file-level classifier


# --- Definition of High-Risk Java API Packages ---
# These are packages known to be complex and potential sources of bugs (I/O, reflection, concurrency, etc.)
HIGH_RISK_API_KEYWORDS = [
    "java.io",
    "java.nio",
    "java.net",
    "java.sql",
    "java.rmi",
    "java.lang.reflect",
    "java.lang.Thread",
    "java.util.concurrent",
    "javax.xml",
    "org.xml.sax",
]

# --- Definitions for Cognitive Complexity ---
# Operators that increment cognitive complexity
LOGICAL_OPERATORS = {"&&": 1, "||": 1, "&": 1, "|": 1, "^": 1}
# Keywords that increment nesting level and complexity
NESTING_KEYWORDS = {"if": 1, "for": 1, "while": 1, "catch": 1, "switch": 1}
# Keywords that break linear flow but don't add nesting
BREAK_KEYWORDS = {"else": 1, "goto": 1, "break": 1, "continue": 1}

# --- Definitions for State Change Count ---
ASSIGNMENT_OPERATORS = [
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
    ">>>=",
]
INCREMENT_DECREMENT_OPERATORS = ["++", "--"]


def calculate_cognitive_complexity(line: str) -> int:
    """
    Calculates a simplified Cognitive Complexity score for a single line of code.
    This function increments complexity for logical operators and nesting keywords.
    """
    score = 0
    tokens = line.split()
    # Penalize for logical operators
    for op in LOGICAL_OPERATORS:
        score += line.count(op) * LOGICAL_OPERATORS[op]

    # Penalize for nesting and flow-breaking keywords
    for token in tokens:
        if token in NESTING_KEYWORDS:
            score += NESTING_KEYWORDS[token]
        if token in BREAK_KEYWORDS:
            score += BREAK_KEYWORDS[token]

    # A simple proxy for nesting: count leading spaces/tabs
    # Each 4 spaces or 1 tab contributes to the nesting level
    leading_spaces = len(line) - len(line.lstrip(" "))
    nesting_level = leading_spaces // 4
    score += nesting_level

    return score


def calculate_api_weighted_nfc(line: str) -> int:
    """
    Calculates an API-Weighted Number of Function Calls (AW-NFC).
    It starts with a base count of function calls and applies a multiplier
    if the line contains keywords from high-risk API packages.
    """
    base_nfc = line.count("(")

    # Check for high-risk API usage
    is_high_risk = any(keyword in line for keyword in HIGH_RISK_API_KEYWORDS)

    # Apply a multiplier (e.g., 2x) if it's a high-risk line
    multiplier = 2 if is_high_risk else 1

    return base_nfc * multiplier


def calculate_state_change_count(line: str) -> int:
    """
    Calculates the State Change Count (SCC) by counting assignment
    and increment/decrement operators.
    """
    score = 0
    for op in ASSIGNMENT_OPERATORS:
        score += line.count(op)
    for op in INCREMENT_DECREMENT_OPERATORS:
        score += line.count(op)
    return score


# --- THE GLANCE++ MODEL ---
class GlancePlus(BaseModel):
    """
    The GLANCE++ model with the following improvements::
    1. A powerful LightGBM classifier trained on the original LEXICAL-ONLY features for file-level prediction.
    2. A set of enhanced, semantically-aware metrics for line-level prediction.
    """

    model_name = "GlancePlus"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test=False,
        is_realistic=False,
    ):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        # --- Components for File-Level Prediction ---
        self.vectorizer = TfidfVectorizer(lowercase=False, min_df=3, max_features=5000)
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

        # --- Components for Line-Level Prediction ---
        self.line_threshold = line_threshold
        self.tokenizer = (
            self.vectorizer.build_tokenizer()
        )  # Use the same tokenizer for consistency
        self.w_ccs = 1.0
        self.w_awnfc = 1.5
        self.w_scc = 1.0

    def file_level_prediction(self):
        """
        Implementation of the file-level classifier using ONLY lexical features.
        """
        print(
            f"Starting file-level prediction for {self.test_release} using {self.model_name}"
        )
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        # 1. Create the lexical feature matrix for training data
        X_train = self.vectorizer.fit_transform(self.train_text)
        y_train = self.train_label

        # 2. Create the lexical feature matrix for test data
        X_test = self.vectorizer.transform(self.test_text)

        print("Training LightGBM model on lexical-only features...")
        self.clf.fit(X_train, y_train)
        print("Training complete.")

        self.test_pred_labels = self.clf.predict(X_test)
        self.test_pred_scores = self.clf.predict_proba(X_test)[:, 1]
        self.save_file_level_result()
        print("File-level prediction complete.")

    def line_level_prediction(self):
        """
        Implementation of the enhanced line-level metrics.
        """
        super().line_level_prediction()
        if USE_CACHE and os.path.exists(self.line_level_result_file):
            return

        predicted_lines, predicted_score, predicted_density = [], [], []
        defective_file_index = [
            i for i, label in enumerate(self.test_pred_labels) if label == 1
        ]

        for i in defective_file_index:
            defective_filename = self.test_filename[i]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[i]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index, line in enumerate(defective_file_line_list):
                tokens = self.tokenizer(line)

                ccs = calculate_cognitive_complexity(line)
                aw_nfc = calculate_api_weighted_nfc(line)
                scc = calculate_state_change_count(line)

                score = (
                    (self.w_ccs * ccs)
                    + (self.w_awnfc * aw_nfc)
                    + (self.w_scc * scc)
                    + 1
                )
                hit_count[line_index] = score

                if any(
                    k in tokens
                    for k in NESTING_KEYWORDS or BREAK_KEYWORDS or ["return"]
                ):
                    cc_count[line_index] = True

            sorted_index = np.argsort(hit_count).tolist()[::-1][
                : int(num_of_lines * self.line_threshold)
            ]
            sorted_index = [idx for idx in sorted_index if hit_count[idx] > 1]

            resorted_index = [idx for idx in sorted_index if cc_count[idx]]
            resorted_index.extend([idx for idx in sorted_index if not cc_count[idx]])

            predicted_score.extend([hit_count[idx] for idx in resorted_index])
            predicted_lines.extend(
                [f"{defective_filename}:{idx + 1}" for idx in resorted_index]
            )
            density = f"{len(np.where(hit_count > 1)[0]) / num_of_lines if num_of_lines > 0 else 0}"
            predicted_density.extend([density for _ in resorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()


# --- THE GLANCE++ MODEL WITH HYBRID FEATURE SET ---
class GlancePlus_Hybrid(BaseModel):
    """
    The GLANCE++ model with the following improvements:
    1. A powerful LightGBM classifier with a hybrid feature set for file-level prediction.
    2. A set of enhanced, semantically-aware metrics for line-level prediction.
    """

    model_name = "GlancePlus-Hybrid"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test=False,
        is_realistic=False,
    ):
        super().__init__(train_release, test_release, is_realistic=is_realistic)

        # --- Components for File-Level Prediction (from Task 2) ---
        self.vectorizer = TfidfVectorizer(lowercase=False, min_df=3, max_features=5000)
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

        # --- Components for Line-Level Prediction ---
        self.line_threshold = line_threshold
        self.tokenizer = self.vector.build_tokenizer()
        self.w_ccs = 1.0
        self.w_awnfc = 1.5
        self.w_scc = 1.0

    def _create_hybrid_feature_matrix(
        self, release_name, file_list, text_data, fit_vectorizer=False
    ):
        """
        Helper function to create the hybrid feature matrix for the file-level classifier.
        """
        traditional_features_df = extract_hybrid_features(release_name, file_list)
        if fit_vectorizer:
            lexical_features = self.vectorizer.fit_transform(text_data)
        else:
            lexical_features = self.vectorizer.transform(text_data)
        aligned_traditional_features = traditional_features_df.reindex(
            file_list
        ).fillna(0)
        traditional_features_sparse = csr_matrix(aligned_traditional_features.values)
        hybrid_features = hstack([lexical_features, traditional_features_sparse])
        return hybrid_features

    def file_level_prediction(self):
        """
        Implementation of the enhanced file-level classifier (Task 2).
        """
        print(
            f"Starting file-level prediction for {self.test_release} using {self.model_name}"
        )
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        X_train = self._create_hybrid_feature_matrix(
            self.train_release,
            self.train_filename,
            self.train_text,
            fit_vectorizer=True,
        )
        y_train = self.train_label
        X_test = self._create_hybrid_feature_matrix(
            self.test_release, self.test_filename, self.test_text
        )

        print("Training LightGBM model on hybrid features...")
        self.clf.fit(X_train, y_train)
        print("Training complete.")

        self.test_pred_labels = self.clf.predict(X_test)
        self.test_pred_scores = self.clf.predict_proba(X_test)[:, 1]
        self.save_file_level_result()
        print("File-level prediction complete.")

    def line_level_prediction(self):
        """
        Implementation of the enhanced line-level metrics (Task 1).
        """
        super().line_level_prediction()  # Call parent to handle path setup
        if USE_CACHE and os.path.exists(self.line_level_result_file):
            return

        predicted_lines, predicted_score, predicted_density = [], [], []
        defective_file_index = [
            i for i, label in enumerate(self.test_pred_labels) if label == 1
        ]

        for i in defective_file_index:
            defective_filename = self.test_filename[i]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[i]

            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=float)
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index, line in enumerate(defective_file_line_list):
                tokens = self.tokenizer(line)

                ccs = calculate_cognitive_complexity(line)
                aw_nfc = calculate_api_weighted_nfc(line)
                scc = calculate_state_change_count(line)

                score = (
                    (self.w_ccs * ccs)
                    + (self.w_awnfc * aw_nfc)
                    + (self.w_scc * scc)
                    + 1
                )
                hit_count[line_index] = score

                if any(
                    k in tokens
                    for k in NESTING_KEYWORDS or BREAK_KEYWORDS or ["return"]
                ):
                    cc_count[line_index] = True

            sorted_index = np.argsort(hit_count).tolist()[::-1][
                : int(num_of_lines * self.line_threshold)
            ]
            sorted_index = [idx for idx in sorted_index if hit_count[idx] > 1]

            resorted_index = [idx for idx in sorted_index if cc_count[idx]]
            resorted_index.extend([idx for idx in sorted_index if not cc_count[idx]])

            predicted_score.extend([hit_count[idx] for idx in resorted_index])
            predicted_lines.extend(
                [f"{defective_filename}:{idx + 1}" for idx in resorted_index]
            )
            density = f"{len(np.where(hit_count > 1)[0]) / num_of_lines if num_of_lines > 0 else 0}"
            predicted_density.extend([density for _ in resorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.save_line_level_result()
        self.save_buggy_density_file()
