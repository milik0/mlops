import joblib
import numpy as np
import subprocess
import os

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
    
    # Générer le code C
    c_code = f"""#include <stdio.h>

// Fonction de prédiction générée automatiquement
double prediction(double *features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Erreur: nombre de features incorrect (attendu: {n_features}, reçu: %d)\\n", n_features);
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

// Fonction main pour tester
int main() {
    // Données de test (à adapter selon vos besoins)
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
    
    print(f"\n✓ Code C généré dans {output_c_file}")
    
    # display compilation commands
    compile_cmd = f"gcc {output_c_file} -o model -lm"
    print(f"\nCommande de compilation : {compile_cmd}")
    
    try:
        print("\nCompilation en cours...")
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Compilation réussie!")
            print("\nPour exécuter le programme : ./model")
            
            # execution of the C program
            print("\nExécution du programme C :")
            print("-" * 50)
            exec_result = subprocess.run("./model", shell=True, capture_output=True, text=True)
            print(exec_result.stdout)
            
            # verification with python program
            print("\nVérification avec le modèle Python :")
            print("-" * 50)
            python_pred = model.predict([test_features])[0]
            print(f"Prédiction Python: {python_pred}")
            
        else:
            print(f"✗ Erreur de compilation : {result.stderr}")
    
    except Exception as e:
        print(f"✗ Erreur lors de la compilation : {e}")
        print(f"Vous pouvez compiler manuellement avec : {compile_cmd}")
    
    return model

if __name__ == "__main__":
    # transpiling the model
    model = transpile_linear_regression("regression.joblib", "model.c")