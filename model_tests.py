import pickle
import pandas as pd

## Load Model
with open("iris_gnb_model.pkl", "rb") as f:
    saved_model = pickle.load(f)


model = saved_model["model"]
X_test_scaled = saved_model["X_test_scaled.csv"]
print(model.predict(X_test_scaled))


if __name__ == "__main__":
    main()