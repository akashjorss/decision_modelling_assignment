from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus

from nutriscore.constants import ATTRIBUTES, GRADE_HIERARCHY, HARMFUL_ATTRIBUTES, PETALES
from nutriscore.utils import data_reader, preference_reader


class AdditiveScore:
    def __init__(self, df: pd.DataFrame, preference_df: pd.DataFrame, start: int = 0, end: int = 20):
        self._start = start
        self._end = end
        self._problem_variables = []
        self._epsilons = []
        self._marginal_utility_variables_mapping: Dict[str, LpVariable] = {}
        self._prob = LpProblem("The Nutriscore", LpMaximize)
        self.df = validation(df, "input")
        self.preference_df = validation(preference_df, "preference")

    def compute_additive_score(self) -> pd.DataFrame:
        self._compute_marginal_utility()
        return self.df

    def _compute_marginal_utility(self) -> Dict[str, float]:
        self._epsilons, self._prob = self._construct_objective()
        self._prob, self._problem_variables = self._construct_problem_variables(self.df, type="input")
        self._prob = self._construct_reference_preference()
        self._prob = self._construct_attribute_monotonicity()
        self._prob.writeLP("The Nutriscore.lp")
        self._prob.solve()
        our_scores = []
        for problem_variable in self._problem_variables:
            our_scores.append(problem_variable.varValue)
        self.df["our_score"] = our_scores
        print("Status:", LpStatus[self._prob.status])
        self.df.to_csv("additive_model_score_results.csv")
        self._plot_attributes()
        return {}

    def _construct_problem_variables(self, df: pd.DataFrame, type: str) -> (LpProblem, List[LpVariable]):
        """
        Construct U(a) = u1(attr1_value) + u2(attr2_value) + ... + u6(attr6_value)
        """
        problem_variables = []
        for index, row in df.iterrows():
            problem_variable = LpVariable(f"product_{type}_{index}_problem", self._start, self._end)
            problem_variables.append(problem_variable)
            attribute_values = []
            for attribute in ATTRIBUTES:
                attribute_value = row[attribute]
                key_name = construct_marginal_value_key(attribute, attribute_value)
                attribute_values.append(
                    self._marginal_utility_variables_mapping.setdefault(
                        key_name, LpVariable(key_name, self._start, self._end)
                    )
                )
            summed_attributes = attribute_values[0]
            for i in range(1, len(attribute_values)):
                summed_attributes += attribute_values[i]
            self._prob += (summed_attributes == problem_variable), f"product_{type}_{index}_constraint"
        return self._prob, problem_variables

    def _construct_reference_preference(self) -> LpProblem:
        """
        Evenly sample 30% of data across all grades from the input dataset to be used as reference set.
        For the reference set add constraints such that the problem variable U(a) for class with higher grades have values
        bigger than that of lower grades by atleast epsilon of the higher grade.
        [a] >= [b, c, d, e] + epsilon(a); [b] >= [c, d, e] + epsilon(b); [c] >= [d, e] + epsilon(c); [d] >= [e] + epsilon(d)
        """
        _, preference_df_problem_variables = self._construct_problem_variables(self.preference_df, type="preference")
        for i in range(0, len(GRADE_HIERARCHY) - 1):
            filtered_i = self.preference_df[self.preference_df["nutriscoregrade"] == GRADE_HIERARCHY[i]]
            filtered_j = self.preference_df[self.preference_df["nutriscoregrade"] == GRADE_HIERARCHY[i + 1]]
            for f_i in filtered_i.index:
                for f_j in filtered_j.index:
                    self._prob += (
                        preference_df_problem_variables[f_j] + self._epsilons[i]
                        <= preference_df_problem_variables[f_i],
                        f"product_{f_i} better than product_{f_j}",
                    )
        return self._prob

    def _construct_attribute_monotonicity(self) -> LpProblem:
        seen_constraints = []
        for attribute in ATTRIBUTES:
            sorted_values = sorted(self.df[attribute].values, reverse=attribute in HARMFUL_ATTRIBUTES)
            for i in range(len(sorted_values) - 1):
                lkey = construct_marginal_value_key(attribute, sorted_values[i])
                rkey = construct_marginal_value_key(attribute, sorted_values[i + 1])
                if lkey == rkey:
                    continue
                if (lkey, rkey) in seen_constraints:
                    continue
                self._prob += (
                    self._marginal_utility_variables_mapping[lkey] <= self._marginal_utility_variables_mapping[rkey],
                    f"{rkey} more preferred than {lkey}",
                )
                seen_constraints.append((lkey, rkey))
        return self._prob

    def _construct_objective(self) -> (List[LpVariable], LpProblem):
        """
        Construct optimization constraint using epsilons to represent gap between nutri score grades
        """
        start = self._start if self._start > 0 else 1
        epsilon_a = LpVariable("epsilon_a", start, self._end)
        epsilon_b = LpVariable("epsilon_b", start, self._end)
        epsilon_c = LpVariable("epsilon_c", start, self._end)
        epsilon_d = LpVariable("epsilon_d", start, self._end)
        self._prob += (
            epsilon_a + epsilon_b + epsilon_c + epsilon_d,
            "slack variables (differences between two consecutive classes) to be maximized",
        )
        epsilons = [epsilon_a, epsilon_b, epsilon_c, epsilon_d]
        return epsilons, self._prob

    def _plot_attributes(self):
        for attribute in ATTRIBUTES:
            values_df = self.df[[attribute]]
            variable_values = []
            for value in values_df[attribute].values:
                variable_values.append(
                    self._marginal_utility_variables_mapping[construct_marginal_value_key(attribute, value)].varValue
                )
            values_df["marginal_utility_value"] = variable_values
            values_df.plot(kind="scatter", x=attribute, y="marginal_utility_value")
            plt.savefig(f"{attribute}_mariginal_utility_plot.png")


def validation(df: pd.DataFrame, type: str) -> pd.DataFrame:
    required_attributes = ["nutriscoregrade"]
    required_attributes.extend(ATTRIBUTES)
    for required_attribute in required_attributes:
        if required_attribute not in df.columns:
            raise AssertionError(f"Attribute {required_attributes} required but not present in {type}: {df.columns}")
    df["nutriscoregrade"] = df["nutriscoregrade"].str.lower()
    for attribute in ATTRIBUTES:
        df[attribute] = df[attribute].fillna(0)
    return df


def construct_marginal_value_key(attribute: str, attribute_value: float):
    key_name = f"{attribute}_{attribute_value}"
    return key_name


def main():
    df = data_reader(PETALES)
    preference_df = preference_reader(PETALES)
    AdditiveScore(df, preference_df).compute_additive_score()


if __name__ == "__main__":
    main()
