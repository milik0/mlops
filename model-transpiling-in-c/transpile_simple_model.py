import joblib
import numpy as np
import subprocess
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def transpile_linear_regression(model_path, output_c_file="model.c"):
    """
    Loading linear regression model and generate the equivalent in C
    """
    # loading the model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # coefficients and intercepts
    coefficients = model.coef_
    intercept = model.intercept_
    n_features = len(coefficients)
    
    print(f"Modèle chargé : {n_features} features")
    print(f"Coefficients : {coefficients}")
    print(f"Intercept : {intercept}")
    
    # code for prediction
    c_code = f"""#include <stdio.h>

// prediction function
double prediction(double *features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Error: nb features incorrect (expected: {n_features}, got: %d)\\n", n_features);
        return -1.0;
    }}
    
    double result = {intercept};
"""
    
    # adding every coef
    for i, coef in enumerate(coefficients):
        c_code += f"    result += {coef} * features[{i}];  // feature {i}\n"
    
    c_code += """    
    return result;
}

int main() {
    // test data
"""
    
    # test example
    test_features = [100000.0, 2.0, 0.0]  # size, nb_rooms, garden
    
    c_code += f"    double test_data[] = {{{', '.join([str(x) for x in test_features])}}};\n"
    c_code += f"    int n_features = {n_features};\n"
    c_code += """    
    double pred = prediction(test_data, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%f", test_data[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
    printf("Prediction: %.8f\\n", pred);
    
    return 0;
}
"""
    
    with open(output_c_file, 'w') as f:
        f.write(c_code)
    
    print(f"\nCode C généré dans {output_c_file}")
    
    # display compilation commands
    compile_cmd = f"gcc {output_c_file} -o model -lm"
    print(f"\nCompilation command : {compile_cmd}")
    
    try:
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            
            # execution of the C program
            print("\nExecution of C program :")
            print("-" * 50)
            exec_result = subprocess.run("./model", shell=True, capture_output=True, text=True)
            print(exec_result.stdout)
            
            # verification with python program
            print("\nVerification with python :")
            print("-" * 50)
            python_pred = model.predict([test_features])[0]
            print(f"Prediction with python: {python_pred}")
            
        else:
            print(f"Error in compilation : {result.stderr}")
    
    except Exception as e:
        print(f"Error in compilation : {e}")
    
    return model

def transpile_logistic_regression(model_path, output_c_file="model.c"):
    """
    Loading logistic regression model and generate the equivalent in C
    """
    print(f"Loading logistic regression model from {model_path}...")
    model = joblib.load(model_path)
    
    coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
    n_features = len(coefficients)
    
    print(f"Loaded model : {n_features} features")
    print(f"Coefficients : {coefficients}")
    print(f"Intercept : {intercept}")
    
    c_code = f"""#include <stdio.h>
#include <math.h>

// Sigmoid function
double sigmoid(double x) {{
    return 1.0 / (1.0 + exp(-x));
}}

double predict_proba(double *features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Error: nb features incorrect (expected: {n_features}, got: %d)\\n", n_features);
        return -1.0;
    }}
    
    double logit = {intercept};
"""
    
    for i, coef in enumerate(coefficients):
        c_code += f"    logit += {coef} * features[{i}];  // feature {i}\n"
    
    c_code += """    
    return sigmoid(logit);
}

int predict_class(double *features, int n_features) {
    double proba = predict_proba(features, n_features);
    return proba >= 0.5 ? 1 : 0;
}

int main() {
    // Test data
"""
    
    test_features = [0.0] * n_features
    
    c_code += f"    double test_data[] = {{{', '.join([str(x) for x in test_features])}}};\n"
    c_code += f"    int n_features = {n_features};\n"
    c_code += """    
    double proba = predict_proba(test_data, n_features);
    int class_pred = predict_class(test_data, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%f", test_data[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
    printf("Probability: %.8f\\n", proba);
    printf("Predicted Class: %d\\n", class_pred);
    
    return 0;
}
"""
    
    with open(output_c_file, 'w') as f:
        f.write(c_code)
    
    print(f"\nCode C généré dans {output_c_file}")
    compile_and_run(output_c_file, model, test_features)
    return model

def transpile_decision_tree(model_path, output_c_file="model.c"):
    """
    Loading decision tree model and generate the equivalent in C
    """
    print(f"Loading decision tree model from {model_path}...")
    model = joblib.load(model_path)
    
    tree = model.tree_
    n_features = tree.n_features
    
    print(f"Loaded model : {n_features} features, {tree.node_count} nodes")
    
    # generating code in C for traversal
    def generate_tree_code(node_id=0, depth=0):
        indent = "    " * (depth + 1)
        
        if tree.feature[node_id] != -2:  # not leaf
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            code = f"{indent}if (features[{feature_idx}] <= {threshold}) {{\n"
            code += generate_tree_code(left_child, depth + 1)
            code += f"{indent}}} else {{\n"
            code += generate_tree_code(right_child, depth + 1)
            code += f"{indent}}}\n"
            return code
        else:
            value = tree.value[node_id][0]
            if len(value) == 1:  # Regression
                prediction = value[0]
                return f"{indent}return {prediction};\n"
            else:  # Classification
                prediction = np.argmax(value)
                return f"{indent}return {prediction};\n"
    
    is_classifier = isinstance(model, DecisionTreeClassifier)
    
    c_code = f"""#include <stdio.h>

double prediction(double *features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Error: incorrect nb of features (expected: {n_features}, got: %d)\\n", n_features);
        return -1.0;
    }}
    
{generate_tree_code()}}}

int main() {{
    // test data
"""
    
    test_features = [1.0] * n_features
    
    c_code += f"    double test_data[] = {{{', '.join([str(x) for x in test_features])}}};\n"
    c_code += f"    int n_features = {n_features};\n"
    c_code += """    
    double pred = prediction(test_data, n_features);
    
    printf("features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%f", test_data[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
"""
    
    if is_classifier:
        c_code += """    printf("Predicted Class: %d\\n", (int)pred);
"""
    else:
        c_code += """    printf("Prediction: %.8f\\n", pred);
"""
    
    c_code += """    
    return 0;
}
"""
    
    with open(output_c_file, 'w') as f:
        f.write(c_code)
    
    print(f"\nCode in C generated in {output_c_file}")
    compile_and_run(output_c_file, model, test_features)
    return model

def compile_and_run(output_c_file, model, test_features):
    compile_cmd = f"gcc {output_c_file} -o model -lm"
    print(f"\nCompilation Command : {compile_cmd}")
    
    try:
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\nExecution of C program:")
            print("-" * 50)
            exec_result = subprocess.run("./model", shell=True, capture_output=True, text=True)
            print(exec_result.stdout)
            
            print("\nVerification with python model:")
            print("-" * 50)
            python_pred = model.predict([test_features])
            if hasattr(model, 'predict_proba'):
                python_proba = model.predict_proba([test_features])[0]
                print(f"Prediction in python (classe): {python_pred[0]}")
                print(f"Prediction in python (proba): {python_proba}")
            else:
                print(f"Prediction in python: {python_pred[0]}")
            
        else:
            print(f"Error in compilation : {result.stderr}")
    
    except Exception as e:
        print(f"Error in compilation : {e}")

def transpile_model(model_path, output_c_file="model.c"):
    """
    Auto-detect model type and transpile accordingly
    """
    model = joblib.load(model_path)
    model_type = type(model).__name__
    
    print(f"Detected model type: {model_type}")
    
    if isinstance(model, LinearRegression):
        return transpile_linear_regression(model_path, output_c_file)
    elif isinstance(model, LogisticRegression):
        return transpile_logistic_regression(model_path, output_c_file)
    elif isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return transpile_decision_tree(model_path, output_c_file)
    else:
        raise ValueError(f"Model type {model_type} is not supported. Supported types: LinearRegression, LogisticRegression, DecisionTreeClassifier")

if __name__ == "__main__":
    # auto-detect and transpile the model
    model = transpile_model("logistic.joblib", "model.c")