import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["distress_score"] = (
        data["stress_level"]
        + data["burnout_score"]
        + data["depression_score"]
    ) / 3

    data["mental_wellness"] = (
        data["mental_health_index"] - data["distress_score"]
    )

    data["stress_depression"] = (
        data["stress_level"] * data["depression_score"]
    )

    data["lifestyle_score"] = (
        data["sleep_hours"] * data["physical_activity"]
    )

    data["academic_score"] = (
        data["study_hours_per_day"] * data["academic_performance"]
    )

    data["social_pressure"] = (
        data["exam_pressure"]
        + data["anxiety_score"]
        + data["financial_stress"]
        + data["family_expectation"]
        - data["social_support"]
    )

    data["academic_social_ratio"] = (
        data["academic_score"] / (data["social_pressure"] + 1)
    )

    data["stress_anxiety"] = (
        data["stress_level"] * data["anxiety_score"]
    )

    data["net_wellbeing"] = (
        data["mental_health_index"]
        - data["stress_level"]
        - data["anxiety_score"]
    )

    return data