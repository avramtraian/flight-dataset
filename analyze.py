import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import config

TARGET_COLUMN = "is_delayed"
DEFAULT_DPI = 300

MONTH_TO_NUMBER = {
    config.Month.JANUARY.value: 0,
    config.Month.FEBRUARY.value: 1,
    config.Month.MARCH.value: 2,
    config.Month.APRIL.value: 3,
    config.Month.MAY.value: 4,
    config.Month.JUNE.value: 5,
    config.Month.JULY.value: 6,
    config.Month.AUGUST.value: 7,
    config.Month.SEPTEMBER.value: 8,
    config.Month.OCTOBER.value: 9,
    config.Month.NOVEMBER.value: 10,
    config.Month.DECEMBER.value: 11,
}

WEATHER_TO_NUMBER = {
    config.Weather.CLEAR.value: 0,
    config.Weather.CLOUDY.value: 1,
    config.Weather.RAIN.value: 2,
    config.Weather.FOG.value: 3,
    config.Weather.SNOW.value: 4,
    config.Weather.RAINSTORM.value: 5,
    config.Weather.THUNDERSTORM.value: 6,
}

def analysis_missing_values(df: pd.DataFrame, file_path: str):
    labels = list()
    missing_percentages = list()
    for column in df.columns:
        percentage = df[column].isna().mean() * 100
        if percentage != 0:
            labels.append(column)
            missing_percentages.append(percentage)

    plt.figure(figsize=(13, 5))
    plt.bar(labels, missing_percentages)
    plt.ylabel("Lipsa (%)")
    plt.title("Procentul de date lipsa")
    plt.savefig(file_path, dpi=DEFAULT_DPI)
    plt.close()

def analysis_numeric_distributions(df: pd.DataFrame, numeric_columns: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, column in enumerate(numeric_columns):
        sb.histplot(df[column].dropna(), kde=True)
        plt.title(column)
        plt.ylabel("Frecventa")
        plt.tight_layout()
        plt.savefig(output_dir + "/" + column + ".png", dpi=DEFAULT_DPI)
        plt.close()

def analysis_categorical_distributions(
    df: pd.DataFrame,
    categorical_columns: list[str],
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    for i, column in enumerate(categorical_columns):
        plt.figure(figsize=(13, 5))
        sb.countplot(x=df[column])
        plt.title(column)
        plt.ylabel("Numar zboruri")
        plt.xlabel(None)
        plt.tight_layout()
        plt.savefig(output_dir + "/" + column + ".png", dpi=DEFAULT_DPI)
        plt.close()

def analysis_outlier_boxplots(df: pd.DataFrame, numeric_columns: list[str], output_filepath: str):
    plt.figure(figsize=(10, 5))
    for i, column in enumerate(numeric_columns):
        plt.subplot(1, len(numeric_columns), i + 1)
        sb.boxplot(y=df[column])
        plt.title(column)
        plt.xlabel(None)
        plt.ylabel(None)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=DEFAULT_DPI)
    plt.close()

def analysis_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str], output_filepath: str):
    plt.figure(figsize=(8, 6))
    sb.heatmap(df[numeric_columns].corr(), cmap="coolwarm")
    plt.title("Matrice de corelatii")
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=DEFAULT_DPI)
    plt.close()

def analysis_numeric_vs_target(df: pd.DataFrame, numeric_columns: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, column in enumerate(numeric_columns):
        plt.figure(figsize=(10, 5))
        sb.violinplot(x=df["is_delayed"], y=df[column])
        plt.title(column)
        plt.tight_layout()
        plt.savefig(output_dir + "/" + column + ".png", dpi=DEFAULT_DPI)
        plt.close()

def main():
    df = pd.read_csv("train.csv")
    os.makedirs("figs", exist_ok=True)

    NUMERIC_COLUMNS = [
        "departure_congestion",
        "arrival_congestion",
        "ticket_price",
    ]

    CATEGORICAL_COLUMNS = [
        "month",
        "departure_airport",
        "arrival_airport",
        "airline",
        "departure_weather",
        "arrival_weather",
        "is_delayed",
    ]

    analysis_missing_values(df, "figs/missing_fields.png");
    analysis_numeric_distributions(df, NUMERIC_COLUMNS, "figs/numeric_histograms")
    analysis_categorical_distributions(df, CATEGORICAL_COLUMNS, "figs/categorical_countplots")
    analysis_outlier_boxplots(df, NUMERIC_COLUMNS, "figs/outliers.png")

    df["month"] = df["month"].str.strip().map(MONTH_TO_NUMBER)
    df["departure_weather"] = df["departure_weather"].str.strip().map(WEATHER_TO_NUMBER)
    df["arrival_weather"] = df["arrival_weather"].str.strip().map(WEATHER_TO_NUMBER)
    PROCESSED_NUMERIC_COLUMNS = [
        # Default numeric columns:
        "departure_congestion",
        "arrival_congestion",
        "ticket_price",

        # Newly mapped columns:
        "month",
        "departure_weather",
        "arrival_weather",
    ]

    analysis_correlation_heatmap(df, PROCESSED_NUMERIC_COLUMNS, "figs/correlation_heatmap.png")
    analysis_numeric_vs_target(df, PROCESSED_NUMERIC_COLUMNS, "figs/numeric_vs_target")

if __name__ == "__main__":
    main()
