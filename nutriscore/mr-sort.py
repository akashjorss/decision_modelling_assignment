from functools import lru_cache
from typing import Dict, List, Any

import pandas as pd
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from nutriscore.constants import ATTRIBUTES, WEIGHTS, GRADE_HIERARCHY, HARMFUL_ATTRIBUTES, PETALES
from nutriscore.utils import data_reader


@lru_cache(maxsize=1)
def normalized_weights() -> Dict[str, float]:
    total_weight = sum(WEIGHTS.values())
    new_weights = {}
    for i in WEIGHTS.keys():
        new_weights[i] = WEIGHTS[i] / total_weight
    return new_weights


def limiting_profiles(petales: bool) -> pd.DataFrame:
    df = None
    if petales:
        df = pd.read_excel("./data/Profle_OpenFood_Petales.xlsx")
    df.rename(
        columns={
            "1-Energy": "energy100g",
            "3-Satu. fat.": "saturatedfat100g",
            "2-Sugar": "sugars100g",
            "6-Fiber": "fiber100g",
            "5-Protein": "proteins100g",
            "4-Salt": "sodium100g",
        },
        inplace=True,
    )
    return df


def attribute_concordant_index(a_value: float, b_value: float, harmful: bool) -> int:
    if harmful:
        if a_value <= b_value:
            return 1
        else:
            return 0
    else:
        if a_value >= b_value:
            return 1
        else:
            return 0


def does_outrank(a, b, threshold: float = 0.7) -> bool:
    concordant_index = 0
    for attribute in ATTRIBUTES:
        concordant_index += normalized_weights()[attribute] * attribute_concordant_index(
            a[attribute], b[attribute], attribute in HARMFUL_ATTRIBUTES
        )
    return concordant_index >= threshold


def pessimistic_majority_sorting(input_df: pd.DataFrame, profile_df: pd.DataFrame):
    predicted_grades = []
    grades = GRADE_HIERARCHY.copy()
    grades.reverse()
    for i, input_row in input_df.iterrows():
        k = 6
        for _, profile_row in profile_df.iterrows():
            if does_outrank(input_row, profile_row):
                predicted_grades.append(grades[k - 1])
                break
            k = k - 1
    return predicted_grades


def optimistic_majority_sorting(input_df: pd.DataFrame, profile_df: pd.DataFrame):
    predicted_grades = []
    grades = GRADE_HIERARCHY.copy()
    grades.reverse()
    for i, input_row in input_df.iterrows():
        k = -1
        for _, profile_row in profile_df.sort_index(ascending=False).iterrows():
            if does_outrank(profile_row, input_row) and not does_outrank(input_row, profile_row):
                predicted_grades.append(grades[k])
                break
            k = k + 1
    return predicted_grades


def plot(actual: List[Any], predicted: List[Any], type: str):
    multi_class_cm = confusion_matrix(y_target=actual, y_predicted=predicted, binary=False)
    multi_class_plot, ax = plot_confusion_matrix(
        conf_mat=multi_class_cm, class_names=GRADE_HIERARCHY, colorbar=True, show_absolute=True, show_normed=True
    )
    multi_class_plot.suptitle(f"Multi Class Confusion Matrix for {type} Majority Sorting", fontsize=14)
    multi_class_plot.show()


def main():
    df = data_reader(PETALES)
    profiles = limiting_profiles(PETALES)
    pessimistic_predictions = pessimistic_majority_sorting(df, profiles)
    plot(df["nutriscoregrade"].values, pessimistic_predictions, type="Pessimistic")
    optimistic_predictions = optimistic_majority_sorting(df, profiles)
    plot(df["nutriscoregrade"].values, optimistic_predictions, type="Optimistic")
    df["pessimistic_grade"] = pessimistic_predictions
    df["optimistic_grade"] = optimistic_predictions
    df.to_csv("mr-sort_results.csv")


if __name__ == "__main__":
    main(),
