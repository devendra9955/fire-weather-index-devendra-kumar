# program to check fire weather index
import joblib
import numpy as np
import pandas as pd
import random
import sys


try:
    model = joblib.load("Linear_regression_models.pkl")
    scaler = joblib.load("Scalers.pkl")
    print("Model and Scaler loaded successfully!\n")
except Exception as e:
    print("Error loading model or scaler:", e)
    sys.exit()


feature_names = [
    "Temperature", "RH", "Ws", "Rain",
    "FFMC", "DMC", "DC", "ISI", "BUI"
]
print("Using training feature order:", feature_names)


try:
    df = pd.read_csv("Algerian_forest_fires_dataset.csv")
    df.columns = df.columns.str.strip()
    if "Classes" in df.columns:
        df = df.drop(columns=["Classes"])
    df.dropna(inplace=True)
    print(" Dataset loaded successfully!")
except Exception as e:
    print(" Dataset not loaded. Random sample option may not work.")
    df = None


while True:
    print("\n FWI Prediction System ")
    print("1 Enter input manually")
    print("2 Use random test input")
    print("3 Pick random row from dataset")
    print("4 Exit")
    choice = input("Enter your choice (1-4): ").strip()

    user_input = []
    actual_fwi = None

   
    if choice == "1":
        print("\nEnter feature values manually:")
        for col in feature_names:
            while True:
                try:
                    val = float(input(f"Enter {col}: "))
                    user_input.append(val)
                    break
                except ValueError:
                    print(" Please enter a valid number.")


    elif choice == "2":
        print("\n Generating random values...")
        for col in feature_names:
            if col == "Temperature":
                val = random.uniform(10, 40)
            elif col == "RH":
                val = random.uniform(10, 100)
            elif col == "Ws":
                val = random.uniform(0, 20)
            elif col == "Rain":
                val = random.uniform(0, 10)
            elif col in ["FFMC", "DMC", "DC", "ISI", "BUI"]:
                val = random.uniform(0, 150)
           
            user_input.append(val)

        print("\n Generated test values:")
        for n, v in zip(feature_names, user_input):
            print(f"  {n}: {round(v, 2)}")

   
    elif choice == "3":
        if df is None:
            print("Dataset not loaded.")
            continue

        try:
            sample = df.sample(1)
            sample = sample[feature_names + ["FWI"]]
        except KeyError as e:
            print(f"Missing columns: {e}")
            continue

        user_input = [float(x) for x in sample[feature_names].values.flatten()]
        actual_fwi = float(sample["FWI"].values[0])

        print("\nRandom test row:")
        for n, v in zip(feature_names, user_input):
            print(f"  {n}: {round(v, 2)}")

    elif choice == "4":
        print("\n Exiting program. Goodbye!")
        break

    else:
        print(" Invalid choice. Try again.")
        continue

 
    try:
        input_df = pd.DataFrame([user_input], columns=feature_names)
        scaled_input = scaler.transform(input_df)
        predicted_fwi = model.predict(scaled_input)[0]

        print("\n Predicted Fire Weather Index (FWI):", round(predicted_fwi, 3))

        if actual_fwi is not None:
            print(f"Actual FWI from dataset: {round(actual_fwi, 3)}")
            print(f"Difference: {abs(actual_fwi - predicted_fwi):.3f}")

        print("\nPrediction complete!\n")

    except Exception as e:
        print("Error during prediction:", e)
