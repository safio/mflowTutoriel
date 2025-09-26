import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

def show_iris_mapping():
    """Show how features map to iris species"""

    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    print("🌸 IRIS DATASET MAPPING")
    print("=" * 50)

    # Show the class mapping
    print("\n📊 CLASS MAPPING:")
    print("-" * 20)
    for i, species in enumerate(iris.target_names):
        print(f"Class {i} → {species.capitalize()}")

    # Create a combined dataset
    data = X.copy()
    data['species'] = [iris.target_names[target] for target in y]
    data['class_number'] = y

    print(f"\n📈 DATASET OVERVIEW:")
    print("-" * 20)
    print(f"Total samples: {len(data)}")
    print(f"Features: {len(iris.feature_names)}")
    print(f"Classes: {len(iris.target_names)}")

    # Show feature ranges for each species
    print(f"\n🔍 FEATURE RANGES BY SPECIES:")
    print("-" * 35)

    for species in iris.target_names:
        species_data = data[data['species'] == species]
        print(f"\n{species.upper()}:")
        print(f"  Samples: {len(species_data)}")

        for feature in iris.feature_names:
            min_val = species_data[feature].min()
            max_val = species_data[feature].max()
            mean_val = species_data[feature].mean()
            print(f"  {feature}: {min_val:.1f} - {max_val:.1f} cm (avg: {mean_val:.1f})")

    # Show some example mappings
    print(f"\n💡 EXAMPLE FEATURE → SPECIES MAPPINGS:")
    print("-" * 40)

    examples = [
        (0, "Typical small flower"),
        (50, "Medium-sized flower"),
        (100, "Large flower")
    ]

    for idx, description in examples:
        sample = X.iloc[idx]
        species = iris.target_names[y[idx]]
        print(f"\n{description} ({species}):")
        for feature, value in sample.items():
            print(f"  {feature}: {value:.1f} cm")
        print(f"  → Predicted: {species}")

    # Show distinguishing patterns
    print(f"\n🎯 KEY DISTINGUISHING PATTERNS:")
    print("-" * 35)

    setosa = data[data['species'] == 'setosa']
    versicolor = data[data['species'] == 'versicolor']
    virginica = data[data['species'] == 'virginica']

    print(f"\nSETOSA (easiest to identify):")
    print(f"  - Smallest petals: {setosa['petal length (cm)'].mean():.1f} cm length")
    print(f"  - Narrowest petals: {setosa['petal width (cm)'].mean():.1f} cm width")
    print(f"  - Widest sepals: {setosa['sepal width (cm)'].mean():.1f} cm width")

    print(f"\nVERSICOLOR (medium-sized):")
    print(f"  - Medium petals: {versicolor['petal length (cm)'].mean():.1f} cm length")
    print(f"  - Medium petal width: {versicolor['petal width (cm)'].mean():.1f} cm width")

    print(f"\nVIRGINICA (largest):")
    print(f"  - Longest petals: {virginica['petal length (cm)'].mean():.1f} cm length")
    print(f"  - Widest petals: {virginica['petal width (cm)'].mean():.1f} cm width")
    print(f"  - Longest sepals: {virginica['sepal length (cm)'].mean():.1f} cm length")

    # Show the decision rules the model likely uses
    print(f"\n🤖 HOW THE MODEL DECIDES:")
    print("-" * 30)
    print("The model looks at feature combinations to classify:")
    print("1. If petal length < 2.5 cm → SETOSA (almost always)")
    print("2. If petal length 2.5-5.0 cm → VERSICOLOR (usually)")
    print("3. If petal length > 5.0 cm → VIRGINICA (usually)")
    print("4. Petal width helps distinguish between versicolor/virginica")

    return data

def create_feature_visualization():
    """Create a simple visualization of the feature distributions"""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = [iris.target_names[target] for target in iris.target]

    print(f"\n📊 CREATING FEATURE COMPARISON TABLE:")
    print("-" * 40)

    # Create summary statistics
    summary = data.groupby('species').agg({
        'sepal length (cm)': ['mean', 'min', 'max'],
        'sepal width (cm)': ['mean', 'min', 'max'],
        'petal length (cm)': ['mean', 'min', 'max'],
        'petal width (cm)': ['mean', 'min', 'max']
    }).round(1)

    print(summary)

    # Most discriminative features
    print(f"\n🎯 MOST USEFUL FEATURES FOR CLASSIFICATION:")
    print("-" * 45)
    print("1. Petal length (cm) - Best single feature")
    print("2. Petal width (cm) - Second best feature")
    print("3. Sepal length (cm) - Helpful for fine-tuning")
    print("4. Sepal width (cm) - Least discriminative")

    return summary

if __name__ == "__main__":
    # Show the mapping
    data = show_iris_mapping()

    # Show feature comparison
    summary = create_feature_visualization()

    print(f"\n✅ This explains how your model maps measurements to species names!")
    print("   The model learns these patterns from training data.")