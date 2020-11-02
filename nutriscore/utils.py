import pandas as pd


def data_reader(petales: bool) -> pd.DataFrame:
    df = None
    if petales:
        df = pd.read_excel('./data/OpenFood_Petales.xlsx')
    else:
        df = pd.read_excel('./data/openfoodfacts_simplified_database.xlsx')
        df = _clean_own_database(df)
    return df


def preference_reader(petales: bool) -> pd.DataFrame:
    df = None
    if petales:
        df = pd.read_excel('./data/OpenFood_Petales_Preference.xlsx')
    else:
        df = pd.read_excel('./data/openfoodfacts_simplified_database_Preference.xlsx')
        df = _clean_own_database(df)
    return df


def _clean_own_database(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query("nutrition_grade_fr in ['a', 'b', 'c', 'd', 'e']")
    df.rename(columns={'nutrition_grade_fr': 'nutriscoregrade', 'energy_100g': 'energy100g',
                       'saturated-fat_100g': 'saturatedfat100g', 'sugars_100g': 'sugars100g',
                       'fiber_100g': 'fiber100g', 'proteins_100g': 'proteins100g', 'sodium_100g': 'sodium100g'},
              inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df