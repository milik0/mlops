#include <stdio.h>

// Fonction de prédiction générée automatiquement
double prediction(double *features, int n_features) {
    if (n_features != 3) {
        printf("Erreur: nombre de features incorrect (attendu: 3, reçu: %d)\n", n_features);
        return -1.0;
    }
    
    double result = -8152.937710156519;
    result += 717.2583697096838 * features[0];  // feature 0
    result += 36824.195974256305 * features[1];  // feature 1
    result += 101571.84002157034 * features[2];  // feature 2
    
    return result;
}

// Fonction main pour tester
int main() {
    // Données de test (à adapter selon vos besoins)
    double test_data[] = {100000.0, 2.0, 0.0};
    int n_features = 3;
    
    double pred = prediction(test_data, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%f", test_data[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\n");
    printf("Prediction: %.8f\n", pred);
    
    return 0;
}
