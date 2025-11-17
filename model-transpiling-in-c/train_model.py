def build_model():
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    import joblib
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")
    print(f"\n{'='*50}")
    print("EXEMPLE DE PRÉDICTION")
    print(f"{'='*50}")
    test_sample = X.iloc[0:1]
    print(f"Input: {test_sample.values[0]}")
    prediction = model.predict(test_sample)[0]
    actual = y.iloc[0]
    print(f"Prédiction: {prediction:.2f}")
    print(f"Valeur réelle: {actual:.2f}")
    print(f"Erreur: {abs(prediction - actual):.2f}")
    
build_model()