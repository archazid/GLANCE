# src/models/glance_codebert.py

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import re
from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from .glance import (
    Glance,
    Glance_LR,
    _count_function_calls,
)  # We inherit from the base Glance class
from ..utils.helper import root_path  # Import root_path for caching


class Glance_FileBERT(Glance):
    """
    An enhanced GLANCE model that uses CodeBERT for improved file-level
    defect prediction. It overrides the file-level prediction method but
    reuses the line-level logic from the parent Glance class.
    """

    model_name = "Glance-FileBERT"

    def __init__(self, train_release: str = "", test_release: str = ""):
        # Call the parent constructor to load all data and set up paths.
        super().__init__(train_release, test_release)

        # Initialize CodeBERT and a new file-level classifier.
        print("Initializing Glance_FileBERT components...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Glance_FileBERT will run on: {self.device}")

        # Setup for the file embedding cache
        self.cache_dir = os.path.join(root_path, "file_embedding_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created file embedding cache directory at: {self.cache_dir}")

        # Load CodeBERT model and tokenizer
        model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.codebert_model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.codebert_model.eval()

        # Define a new, simple classifier for the file embeddings.
        self.file_level_clf = LogisticRegression(random_state=0, max_iter=1000)

    def _generate_file_embeddings(self, release_name, file_texts):
        """
        Helper function to generate a single CodeBERT embedding for each file.
        Includes caching to avoid re-computation.
        """
        cache_file_path = os.path.join(
            self.cache_dir, f"{release_name}_file_embeddings.pkl"
        )

        if os.path.exists(cache_file_path):
            print(f"Loading file embeddings from cache: {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                return pickle.load(f)

        print(f"No cache found for {release_name}. Generating file embeddings...")
        all_embeddings = []
        with torch.no_grad():
            for text in file_texts:
                # Tokenize the entire file's text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                ).to(self.device)
                # The "pooler_output" is an embedding of the [CLS] token, representing the entire sequence.
                embedding = self.codebert_model(**inputs).pooler_output.cpu().numpy()
                all_embeddings.append(embedding)

        embeddings_array = np.vstack(all_embeddings)

        print(f"Saving newly generated file embeddings to cache: {cache_file_path}")
        with open(cache_file_path, "wb") as f:
            pickle.dump(embeddings_array, f)

        return embeddings_array

    def file_level_prediction(self):
        """
        This method OVERRIDES the file_level_prediction from both BaseModel and Glance_LR.
        It uses CodeBERT embeddings instead of Bag-of-Words or LOC heuristics.
        """
        print(
            f"--- Glance_FileBERT: Predicting Defective Files for {self.test_release} ---"
        )

        # Generate/load embeddings for the training data
        print("Processing training data...")
        train_embeddings = self._generate_file_embeddings(
            self.train_release, self.train_text
        )

        # Train the file-level classifier
        print("Training file-level classifier...")
        self.file_level_clf.fit(train_embeddings, self.train_label)

        # Generate/load embeddings for the testing data
        print("Processing testing data...")
        test_embeddings = self._generate_file_embeddings(
            self.test_release, self.test_text
        )

        # Predict defective files in the test set
        print("Predicting defective files...")
        self.test_pred_labels = self.file_level_clf.predict(test_embeddings)
        self.test_pred_scores = self.file_level_clf.predict_proba(test_embeddings)[:, 1]

        # Save the results using the inherited method from BaseModel
        print("Saving file-level prediction results...")
        self.save_file_level_result()


class Glance_LineBERT(Glance_LR):
    """
    A Semantic Enhancement of GLANCE using CodeBERT.
    Inherits the file-level prediction logic from Glance_LR and
    provides a new, enhanced implementation for line-level prediction.
    """

    model_name = "Glance-LineBERT"

    def __init__(self, train_release: str = "", test_release: str = ""):
        # Call the parent constructor to load all data and set up paths.
        super().__init__(train_release, test_release)

        print(
            "Initializing Glance_LineBERT components (CodeBERT, Fusion Classifier)..."
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Glance_LineBERT will run on: {self.device}")

        # --- Embedding Cache Setup ---
        # Define the path for the embedding cache directory.
        self.cache_dir = os.path.join(root_path, "embedding_cache")
        # Create the directory if it doesn't exist.
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created embedding cache directory at: {self.cache_dir}")
        # --- End Cache Setup ---

        model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.codebert_model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.codebert_model.eval()  # Set model to evaluation mode

        # Heuristic feature normalizer
        self.scaler = MinMaxScaler()

        # Define the new classifier for the fused features.
        # Input size = 3 (GLANCE features) + 768 (CodeBERT embedding) = 771
        self.fusion_classifier = nn.Sequential(
            nn.Linear(771, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def _extract_features_for_release(
        self, release_name, file_texts, file_lines, line_oracle
    ):
        """
        Helper function to extract features and labels for a given release.
        """

        # --- Caching Logic: Check for existing cache file ---
        cache_file_path = os.path.join(self.cache_dir, f"{release_name}_features.pkl")

        if os.path.exists(cache_file_path):
            print(f"Loading features from cache: {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                cached_data = pickle.load(f)
            return (
                cached_data["heuristics"],
                cached_data["semantics"],
                cached_data["labels"],
            )
        # --- End Caching Logic ---

        print(f"No cache found for {release_name}. Generating features on-the-fly...")
        all_heuristic_features = []
        all_semantic_features = []
        all_labels = []

        # Get a set of buggy lines for quick lookup
        buggy_line_set = set()
        for file in line_oracle:
            for line_num in line_oracle[file]:
                buggy_line_set.add(f"{file}:{line_num}")

        # Determine the correct filenames list to use (train or test)
        filenames = (
            self.train_filename
            if release_name == self.train_release
            else self.test_filename
        )

        for i, file_content in enumerate(file_texts):
            filename = filenames[i]
            lines = file_lines[i]

            for j, line_code in enumerate(lines):
                if not line_code.strip():
                    continue

                # Calculate GLANCE heuristic features
                tokens_in_line = self.tokenizer(line_code)
                nt = len(tokens_in_line)
                nfc = _count_function_calls(line_code)
                all_heuristic_features.append(
                    [nt, nfc, 1 if "if" in tokens_in_line else 0]
                )  # Adding CE feature

                # Generate CodeBERT semantic embedding
                with torch.no_grad():
                    inputs = self.tokenizer(
                        line_code,
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    ).to(self.device)
                    outputs = self.codebert_model(**inputs)
                    semantic_embedding = outputs.pooler_output.cpu().numpy().flatten()
                    all_semantic_features.append(semantic_embedding)

                # Get the label
                label = 1 if f"{filename}:{j + 1}" in buggy_line_set else 0
                all_labels.append(label)

        # --- Caching Logic: Save newly generated features ---
        print(f"Saving newly generated features to cache: {cache_file_path}")
        with open(cache_file_path, "wb") as f:
            pickle.dump(
                {
                    "heuristics": np.array(all_heuristic_features),
                    "semantics": np.array(all_semantic_features),
                    "labels": np.array(all_labels),
                },
                f,
            )
        # --- End Caching Logic ---

        return (
            np.array(all_heuristic_features),
            np.array(all_semantic_features),
            np.array(all_labels),
        )

    def line_level_prediction(self):
        """
        This method OVERRIDES the original line-level prediction.
        It first trains the fusion_classifier on the training release,
        then predicts on the test release.
        """

        print("--- Glance_LineBERT: Training Fusion Classifier ---")

        # Extract features from the TRAINING release to train our fusion classifier
        # We need the line-level oracle for the training set for this
        train_line_oracle, _ = self.get_oracle_lines(self.train_release)

        heuristic_train, semantic_train, labels_train = (
            self._extract_features_for_release(
                self.train_release,
                self.train_text,
                self.train_text_lines,
                train_line_oracle,
            )
        )

        # Normalize heuristic features based on the training data
        heuristic_train_scaled = self.scaler.fit_transform(heuristic_train)

        # Fuse features
        fused_train_features = np.concatenate(
            [heuristic_train_scaled, semantic_train], axis=1
        )

        # Train the PyTorch classifier
        optimizer = torch.optim.Adam(self.fusion_classifier.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        # Simple training loop (future research may add epochs, batching, etc.)
        self.fusion_classifier.train()  # Set model to training mode
        inputs = torch.tensor(fused_train_features, dtype=torch.float32).to(self.device)
        targets = (
            torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

        # A single training step for demonstration
        optimizer.zero_grad()
        outputs = self.fusion_classifier(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Fusion classifier training finished. Loss: {loss.item()}")

        print("--- Glance_LineBERT: Predicting on Test Release ---")
        self.fusion_classifier.eval()  # Set model to evaluation mode

        # Extract features from the TEST release for prediction
        heuristic_test, semantic_test, _ = self._extract_features_for_release(
            self.test_release,
            self.test_text,
            self.test_text_lines,
            self.oracle_line_dict,
        )

        # IMPORTANT: Use the scaler fitted on the training data to transform test data
        heuristic_test_scaled = self.scaler.transform(heuristic_test)
        fused_test_features = np.concatenate(
            [heuristic_test_scaled, semantic_test], axis=1
        )

        # Predict scores
        with torch.no_grad():
            inputs = torch.tensor(fused_test_features, dtype=torch.float32).to(
                self.device
            )
            predicted_scores = self.fusion_classifier(inputs).cpu().numpy().flatten()

        # Populate results based on prediction scores
        # Here you would decide on a threshold to classify lines as buggy or not,
        # and then populate self.predicted_buggy_lines and self.predicted_buggy_score
        # For simplicity, we add all lines with their scores and can filter later

        line_counter = 0
        for i, file_content in enumerate(self.test_text):
            filename = self.test_filename[i]
            # Only consider lines in files predicted as defective by Stage 1
            if self.test_pred_labels[i] == 1:
                lines = self.test_text_lines[i]
                for j, line_code in enumerate(lines):
                    if not line_code.strip():
                        continue

                    # The predicted_scores array corresponds to the lines we extracted
                    current_score = predicted_scores[line_counter]
                    self.predicted_buggy_lines.append(f"{filename}:{j + 1}")
                    self.predicted_buggy_score.append(current_score)
                    line_counter += 1

        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)
        self.save_line_level_result()
        self.save_buggy_density_file()


class Glance_FileBERT_Pro(Glance):
    """
    An enhanced GLANCE model that uses CodeBERT for improved file-level
    defect prediction. It overrides the file-level prediction method but
    reuses the line-level logic from the parent Glance class.
    """

    model_name = "Glance-FileBERT"

    def __init__(self, train_release: str = "", test_release: str = ""):
        # Call the parent constructor to load all data and set up paths.
        super().__init__(train_release, test_release)

        # Initialize CodeBERT and a new file-level classifier.
        print("Initializing Glance_FileBERT components...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Glance_FileBERT will run on: {self.device}")

        # Setup for the file embedding cache
        self.cache_dir = os.path.join(root_path, "file_embedding_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created file embedding cache directory at: {self.cache_dir}")

        # Load CodeBERT model and tokenizer
        model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.codebert_model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.codebert_model.eval()

        # Define a new, simple classifier for the file embeddings.
        # A standard Logistic Regression is a great choice for this.
        self.file_level_clf = LogisticRegression(random_state=0, max_iter=1000)

    def _preprocess_text(self, text_lines: list[str]) -> str:
        """
        Applies line-by-line preprocessing to the entire content of a file.
        - Normalizes string and numeric literals.
        - Joins the processed lines back into a single string for CodeBERT.
        """
        processed_lines = []
        for line in text_lines:
            # Normalize string literals
            processed_line = re.sub(r'".*?"', "<STR>", line)
            # Normalize character literals
            processed_line = re.sub(r"'.'", "<CHAR>", processed_line)
            # Normalize all numeric literals
            processed_line = re.sub(r"\b\d+\b", "<NUM>", processed_line)
            processed_lines.append(processed_line)

        # Join the lines back into a single block of text
        return "\n".join(processed_lines)

    def _generate_file_embeddings(self, release_name, file_texts_lines):
        """
        Helper function to generate a single CodeBERT embedding for each file.
        """
        cache_file_path = os.path.join(
            self.cache_dir, f"{release_name}_preprocessed_file_embeddings.pkl"
        )

        if os.path.exists(cache_file_path):
            print(f"Loading preprocessed file embeddings from cache: {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                return pickle.load(f)

        print(
            f"No cache found for {release_name}. Generating preprocessed file embeddings..."
        )
        all_embeddings = []
        with torch.no_grad():
            for lines in file_texts_lines:  # Iterate through the list of line lists
                # --- APPLY THE PREPROCESSING STEP ---
                preprocessed_text = self._preprocess_text(lines)

                # Tokenize the preprocessed text
                inputs = self.tokenizer(
                    preprocessed_text,  # <-- USE THE PREPROCESSED TEXT HERE
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                ).to(self.device)

                # The "pooler_output" is an embedding of the [CLS] token, representing the entire sequence.
                embedding = self.codebert_model(**inputs).pooler_output.cpu().numpy()
                all_embeddings.append(embedding)

        embeddings_array = np.vstack(all_embeddings)

        print(f"Saving newly generated embeddings to cache: {cache_file_path}")
        with open(cache_file_path, "wb") as f:
            pickle.dump(embeddings_array, f)

        return embeddings_array

    def file_level_prediction(self):
        """
        This method OVERRIDES the file_level_prediction from both BaseModel and Glance_LR.
        It uses CodeBERT embeddings instead of Bag-of-Words or LOC heuristics.
        """
        print(
            f"--- Glance_FileBERT: Predicting Defective Files for {self.test_release} ---"
        )

        # Generate/load embeddings for the training data
        print("Processing training data...")
        train_embeddings = self._generate_file_embeddings(
            self.train_release, self.train_text_lines
        )

        # Train the file-level classifier
        print("Training file-level classifier...")
        self.file_level_clf.fit(train_embeddings, self.train_label)

        # Generate/load embeddings for the testing data
        print("Processing testing data...")
        test_embeddings = self._generate_file_embeddings(
            self.test_release, self.test_text_lines
        )

        # Predict defective files in the test set
        print("Predicting defective files...")
        self.test_pred_labels = self.file_level_clf.predict(test_embeddings)
        self.test_pred_scores = self.file_level_clf.predict_proba(test_embeddings)[:, 1]

        # Save the results using the inherited method from BaseModel
        print("Saving file-level prediction results...")
        self.save_file_level_result()


class Glance_LineBERT_Pro(Glance_LR):
    """
    A Semantic Enhancement of GLANCE using CodeBERT.
    Inherits the file-level prediction logic from Glance_LR and
    provides a new, enhanced implementation for line-level prediction.
    """

    model_name = "Glance-LineBERT"

    def __init__(self, train_release: str = "", test_release: str = ""):
        # Call the parent constructor to load all data and set up paths.
        super().__init__(train_release, test_release)

        print(
            "Initializing Glance_LineBERT components (CodeBERT, Fusion Classifier)..."
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Glance_LineBERT will run on: {self.device}")

        # --- Embedding Cache Setup ---
        # Define the path for the embedding cache directory.
        self.cache_dir = os.path.join(root_path, "embedding_cache")
        # Create the directory if it doesn't exist.
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created embedding cache directory at: {self.cache_dir}")
        # --- End Cache Setup ---

        model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.codebert_model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.codebert_model.eval()  # Set model to evaluation mode

        # Heuristic feature normalizer
        self.scaler = MinMaxScaler()

        # Define the new classifier for the fused features.
        # Input size = 3 (GLANCE features) + 768 (CodeBERT embedding) = 771
        self.fusion_classifier = nn.Sequential(
            nn.Linear(771, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def _preprocess_line(self, line_of_code: str) -> str:
        """
        A refined preprocessing function to normalize code lines before
        feeding them to CodeBERT. It focuses on normalizing literals
        while preserving important keywords and structure.
        """
        # Replace string literals with a generic token
        processed_code = re.sub(r'".*?"', "<STR>", line_of_code)
        # Replace character literals
        processed_code = re.sub(r"'.'", "<CHAR>", processed_code)
        # Replace all numeric literals (integers, floats) with a generic token
        processed_code = re.sub(r"\b\d+\b", "<NUM>", processed_code)
        return processed_code

    def _extract_features_for_release(
        self, release_name, file_texts, file_lines, line_oracle
    ):
        """
        Helper function to extract features and labels for a given release.
        """

        # --- Caching Logic: Check for existing cache file ---
        cache_file_path = os.path.join(self.cache_dir, f"{release_name}_features.pkl")

        if os.path.exists(cache_file_path):
            print(f"Loading features from cache: {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                cached_data = pickle.load(f)
            return (
                cached_data["heuristics"],
                cached_data["semantics"],
                cached_data["labels"],
            )
        # --- End Caching Logic ---

        print(f"No cache found for {release_name}. Generating features on-the-fly...")
        all_heuristic_features = []
        all_semantic_features = []
        all_labels = []

        # Get a set of buggy lines for quick lookup
        buggy_line_set = set()
        for file in line_oracle:
            for line_num in line_oracle[file]:
                buggy_line_set.add(f"{file}:{line_num}")

        # Determine the correct filenames list to use (train or test)
        filenames = (
            self.train_filename
            if release_name == self.train_release
            else self.test_filename
        )

        for i, file_content in enumerate(file_texts):
            filename = filenames[i]
            lines = file_lines[i]

            for j, line_code in enumerate(lines):
                if not line_code.strip():
                    continue

                # Calculate GLANCE heuristic features
                tokens_in_line = self.tokenizer(line_code)
                nt = len(tokens_in_line)
                nfc = _count_function_calls(line_code)
                all_heuristic_features.append(
                    [nt, nfc, 1 if "if" in tokens_in_line else 0]
                )  # Adding CE feature

                # Preprocess the line before sending it to CodeBERT
                processed_line = self._preprocess_line(line_code)

                # Generate CodeBERT semantic embedding
                with torch.no_grad():
                    inputs = self.tokenizer(
                        processed_line,
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    ).to(self.device)
                    outputs = self.codebert_model(**inputs)
                    semantic_embedding = outputs.pooler_output.cpu().numpy().flatten()
                    all_semantic_features.append(semantic_embedding)

                # Get the label
                label = 1 if f"{filename}:{j + 1}" in buggy_line_set else 0
                all_labels.append(label)

        # --- Caching Logic: Save newly generated features ---
        print(f"Saving newly generated features to cache: {cache_file_path}")
        with open(cache_file_path, "wb") as f:
            pickle.dump(
                {
                    "heuristics": np.array(all_heuristic_features),
                    "semantics": np.array(all_semantic_features),
                    "labels": np.array(all_labels),
                },
                f,
            )
        # --- End Caching Logic ---

        return (
            np.array(all_heuristic_features),
            np.array(all_semantic_features),
            np.array(all_labels),
        )

    def line_level_prediction(self):
        """
        This method OVERRIDES the original line-level prediction.
        It first trains the fusion_classifier on the training release,
        then predicts on the test release.
        """

        print("--- Glance_LineBERT: Training Fusion Classifier ---")

        # Extract features from the TRAINING release to train our fusion classifier
        # We need the line-level oracle for the training set for this
        train_line_oracle, _ = self.get_oracle_lines(self.train_release)

        heuristic_train, semantic_train, labels_train = (
            self._extract_features_for_release(
                self.train_release,
                self.train_text,
                self.train_text_lines,
                train_line_oracle,
            )
        )

        # Normalize heuristic features based on the training data
        heuristic_train_scaled = self.scaler.fit_transform(heuristic_train)

        # Fuse features
        fused_train_features = np.concatenate(
            [heuristic_train_scaled, semantic_train], axis=1
        )

        # Train the PyTorch classifier
        optimizer = torch.optim.Adam(self.fusion_classifier.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        # Simple training loop (for a thesis, you'd add epochs, batching, etc.)
        self.fusion_classifier.train()  # Set model to training mode
        inputs = torch.tensor(fused_train_features, dtype=torch.float32).to(self.device)
        targets = (
            torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

        # A single training step for demonstration
        optimizer.zero_grad()
        outputs = self.fusion_classifier(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Fusion classifier training finished. Loss: {loss.item()}")

        print("--- Glance_LineBERT: Predicting on Test Release ---")
        self.fusion_classifier.eval()  # Set model to evaluation mode

        # Extract features from the TEST release for prediction
        heuristic_test, semantic_test, _ = self._extract_features_for_release(
            self.test_release,
            self.test_text,
            self.test_text_lines,
            self.oracle_line_dict,
        )

        # IMPORTANT: Use the scaler fitted on the training data to transform test data
        heuristic_test_scaled = self.scaler.transform(heuristic_test)
        fused_test_features = np.concatenate(
            [heuristic_test_scaled, semantic_test], axis=1
        )

        # Predict scores
        with torch.no_grad():
            inputs = torch.tensor(fused_test_features, dtype=torch.float32).to(
                self.device
            )
            predicted_scores = self.fusion_classifier(inputs).cpu().numpy().flatten()

        # Populate results based on prediction scores
        # Here you would decide on a threshold to classify lines as buggy or not,
        # and then populate self.predicted_buggy_lines and self.predicted_buggy_score
        # For simplicity, we add all lines with their scores and can filter later

        line_counter = 0
        for i, file_content in enumerate(self.test_text):
            filename = self.test_filename[i]
            # Only consider lines in files predicted as defective by Stage 1
            if self.test_pred_labels[i] == 1:
                lines = self.test_text_lines[i]
                for j, line_code in enumerate(lines):
                    if not line_code.strip():
                        continue

                    # The predicted_scores array corresponds to the lines we extracted
                    current_score = predicted_scores[line_counter]
                    self.predicted_buggy_lines.append(f"{filename}:{j + 1}")
                    self.predicted_buggy_score.append(current_score)
                    line_counter += 1

        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)
        self.save_line_level_result()
        self.save_buggy_density_file()
