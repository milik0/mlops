from transpile_simple_model import transpile_model

if __name__ == "__main__":
    model = transpile_model("tree.joblib", "model_tree.c")
