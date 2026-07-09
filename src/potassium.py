import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

df = pd.read_excel("potassium.xlsx")     

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())



pca = PCA(n_components=1)

df['PC1'] = pca.fit_transform(df[['R', 'G', 'B']]).flatten()

print("\nPCA Coefficients:")
print(pca.components_)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)


X = df[['PC1']]
y = df['Concentration']


models = {

    "Linear Regression":
        LinearRegression(),

    "Polynomial Regression (Degree 2)":
        make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            objective='reg:squarederror',
            random_state=42
        )
}



for name, model in models.items():

   
    model.fit(X, y)

    # Predict on training data
    y_pred = model.predict(X)


    r2 = r2_score(y, y_pred)

    print(f"\n{name}")
    print(f"Training R² Score: {r2:.4f}")


    sorted_idx = np.argsort(X['PC1'])

    X_sorted = X.iloc[sorted_idx]
    y_sorted = y.iloc[sorted_idx]

    y_pred_sorted = model.predict(X_sorted)


   
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X['PC1'],
        y,
        s=120,
        label='Actual Data'
    )

    plt.plot(
        X_sorted['PC1'],
        y_pred_sorted,
        linewidth=3,
        label='Regression Curve'
    )

    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Concentration (ppm)")
    plt.title(f"{name}\nTraining R² = {r2:.4f}")

    plt.grid(True)
    plt.legend()

    plt.show()



R = float(input("\nEnter Red value: "))
G = float(input("Enter Green value: "))
B = float(input("Enter Blue value: "))

new_rgb = np.array([[R, G, B]])


new_pc1 = pca.transform(new_rgb)


new_pc1_df = pd.DataFrame(
    new_pc1,
    columns=['PC1']
)


print("\nPredicted Concentrations:\n")

for name, model in models.items():

    prediction = model.predict(new_pc1_df)[0]

    # Prevent impossible negative values
    prediction = max(0, prediction)

    print(f"{name}: {prediction:.2f} ppm")