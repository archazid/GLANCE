# -*- coding:utf-8 -*-

import os
import numpy as np

from src.utils.config import USE_CACHE
from src.models.glance import Glance, Glance_LR

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


class GlancePlus_Line(Glance):
    """
    An enhanced version of GLANCE that uses more sophisticated line-level metrics.
    """

    model_name = "GlancePlus-Line"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test_result_path="",
        is_realistic=False,
    ):
        # Initialize the parent Glance class
        super().__init__(
            train_release, test_release, line_threshold, test_result_path, is_realistic
        )
        # Define weights for the new scoring function.
        # These could be tuned via experimentation (e.g., Grid Search on the validation set).
        self.w_ccs = 1.0
        self.w_awnfc = 1.5
        self.w_scc = 1.0

    def line_level_prediction(self):
        """
        This method OVERRIDES the original line_level_prediction from the Glance class.
        It implements the new scoring logic based on our enhanced metrics.
        """
        # This first line is important to ensure paths are set up correctly.
        super(Glance, self).line_level_prediction()
        if USE_CACHE and os.path.exists(self.line_level_result_file):
            return

        predicted_lines, predicted_score, predicted_density = [], [], []

        # Get the files predicted as defective by the file-level classifier (Stage 1)
        defective_file_index = [
            i
            for i in np.argsort(self.test_pred_scores)[::-1]
            if self.test_pred_labels[i] == 1
        ]

        # Iterate through each predicted defective file to rank its lines (Stage 2)
        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]

            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []

            defective_file_line_list = self.test_text_lines[defective_file_index[i]]

            # --- CORE LOGIC OF GLANCE++ ---
            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(
                num_of_lines, dtype=float
            )  # Use float for weighted scores
            cc_count = np.zeros(num_of_lines, dtype=bool)

            for line_index in range(num_of_lines):
                line = defective_file_line_list[line_index]
                tokens = self.tokenizer(line)

                # Calculate our new, enhanced metrics
                ccs = calculate_cognitive_complexity(line)
                aw_nfc = calculate_api_weighted_nfc(line)
                scc = calculate_state_change_count(line)

                # Calculate the new DefectPronenessScore using a weighted sum
                # We add 1 to avoid scores of zero for simple lines
                score = (
                    (self.w_ccs * ccs)
                    + (self.w_awnfc * aw_nfc)
                    + (self.w_scc * scc)
                    + 1
                )
                hit_count[line_index] = score

                # The Control Element (CE) logic remains the same
                if any(
                    keyword in tokens
                    for keyword in [
                        "for",
                        "while",
                        "do",
                        "if",
                        "else",
                        "switch",
                        "case",
                        "continue",
                        "break",
                        "return",
                    ]
                ):
                    cc_count[line_index] = True
            # --- END OF CORE LOGIC ---

            # The ranking logic is identical to the original GLANCE
            sorted_index = np.argsort(hit_count).tolist()[::-1][
                : int(len(hit_count) * self.line_threshold)
            ]
            sorted_index = [
                i for i in sorted_index if hit_count[i] > 1
            ]  # Only consider lines with non-trivial scores

            # Re-sort to bring Control Element lines to the top
            resorted_index = [i for i in sorted_index if cc_count[i]]
            resorted_index.extend([i for i in sorted_index if not cc_count[i]])

            # Store results
            predicted_score.extend([hit_count[i] for i in resorted_index])
            predicted_lines.extend(
                [f"{defective_filename}:{i + 1}" for i in resorted_index]
            )
            density = f"{len(np.where(hit_count > 1)[0]) / num_of_lines if num_of_lines > 0 else 0}"
            predicted_density.extend([density for _ in resorted_index])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        # Save results using the method from BaseModel
        self.save_line_level_result()
        self.save_buggy_density_file()


class GlancePlus_Line_LR(GlancePlus_Line, Glance_LR):
    """
    This class combines the file-level prediction logic of Glance_LR
    with the new line-level prediction logic of GlancePlus.
    """

    model_name = "GlancePlus-Line-LR"

    def __init__(
        self,
        train_release="",
        test_release="",
        line_threshold=0.5,
        test=False,
        is_realistic=False,
    ):
        # We need to initialize both parent classes correctly.
        # Glance_LR handles the file-level prediction.
        # GlancePlus_Line handles the line-level prediction.
        Glance_LR.__init__(
            self, train_release, test_release, line_threshold, test, is_realistic
        )
        GlancePlus_Line.__init__(self, train_release, test_release, line_threshold)
