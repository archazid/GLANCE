# -*- coding:utf-8 -*-
import sys
import time
import pandas as pd
import warnings
from warnings import simplefilter

from src.models.glance import Glance_EA, Glance_LR, Glance_MD
from src.models.glance_plus_file import GlancePlus_File
from src.models.glance_plus_line import GlancePlus_Line_LR
from src.models.glance_plus import GlancePlus
from src.utils.helper import get_project_releases_dict

# Ignore common FutureWarnings from libraries like pandas
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=FutureWarning)

# The model name and its corresponding python class implementation
MODEL_DICT = {
    # GLANCE Models
    "Glance_EA": Glance_EA,
    "Glance_LR": Glance_LR,
    "Glance_MD": Glance_MD,
    # GLANCE++ Models
    "GlancePlus_File": GlancePlus_File,
    "GlancePlus_Line_LR": GlancePlus_Line_LR,
    "GlancePlus": GlancePlus,
}


# ========================= Run RQ1 experiments =================================
def run_cross_release_predict(prediction_model, save_time=False):
    # time
    release_name, build_time_list, pred_time_list = [], [], []
    for project, releases in get_project_releases_dict().items():
        for i in range(len(releases) - 1):
            # 1. Loading data. train data index = i, test data index = i + 1
            print(
                f"========== {prediction_model.model_name} CR PREDICTION for {releases[i + 1]} ================"[
                    :60
                ]
            )
            # ####### Build time #######
            t_start = time.time()
            model = prediction_model(releases[i], releases[i + 1])
            t_end = time.time()
            build_time_list.append(t_end - t_start)

            # ####### Pred time #######
            t_start = time.time()
            model.file_level_prediction()
            model.analyze_file_level_result()
            model.line_level_prediction()
            model.analyze_line_level_result()
            t_end = time.time()
            pred_time_list.append(t_end - t_start)
            release_name.append(releases[i + 1])

            data = {
                "release_name": release_name,
                "build_time": build_time_list,
                "pred_time": pred_time_list,
            }
            data = pd.DataFrame(
                data, columns=["release_name", "build_time", "pred_time"]
            )
            data.to_csv(model.execution_time_file, index=False) if save_time else None


def run_default():
    run_cross_release_predict(Glance_LR)
    run_cross_release_predict(Glance_EA)
    run_cross_release_predict(Glance_MD)
    pass


def parse_args():
    # If there is no additional parameters in the command line, run the default models.
    if len(sys.argv) == 1:
        run_default()
    # Run the specific models.
    else:
        model_name = sys.argv[1]
        if model_name in MODEL_DICT.keys():
            run_cross_release_predict(MODEL_DICT[model_name])


if __name__ == "__main__":
    parse_args()
