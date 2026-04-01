from data_preprocessing import load_and_prepare_data
from train_models import train_and_evaluate
from evaluate import (
    save_results_table,
    plot_class_distribution,
    plot_missing_values,
    plot_model_comparison,
    plot_roc_curves,
    plot_pr_curves,
    plot_threshold_vs_f1,
    plot_probability_distribution,
    plot_confusion_matrices,
    plot_feature_importance,
    save_best_model,
)


def main():
    df, X, y = load_and_prepare_data()
    artifacts = train_and_evaluate(X, y)
    results_df = artifacts["results"]
    trained_models = artifacts["trained_models"]

    best_model_name = results_df.iloc[0]["model"]

    save_results_table(results_df)
    plot_class_distribution(df)
    plot_missing_values(df)
    plot_model_comparison(results_df)
    plot_roc_curves(trained_models)
    plot_pr_curves(trained_models)
    plot_threshold_vs_f1(trained_models)
    plot_probability_distribution(trained_models)
    plot_confusion_matrices(results_df)
    plot_feature_importance(best_model_name, trained_models)
    save_best_model(best_model_name, trained_models)

    print("\n=== MODEL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print("\nAll outputs saved in ./outputs")


if __name__ == "__main__":
    main()