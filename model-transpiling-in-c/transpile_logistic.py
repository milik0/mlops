from transpile_simple_model import transpile_model

if __name__ == "__main__":
    model = transpile_model("logistic.joblib", "model_logistic.c")
