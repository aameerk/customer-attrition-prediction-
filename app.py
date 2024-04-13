try:
    import streamlit as st
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_curve, auc, accuracy_score
    import numpy as np
    from collections import Counter
    import plotly.graph_objects as go
    from river.linear_model import LogisticRegression as RiverLogisticRegression
    from sklearn.model_selection import train_test_split

    # Function to load data
    @st.cache
    def load_data(file):
        data = pd.read_csv(file)
        # Replace empty strings with NaN
        data.replace(' ', np.nan, inplace=True)
        # Drop rows with missing values
        data.dropna(inplace=True)
        return data

    # Function to perform data fusion
    def fusion(accuracy_scores):
        # Soft fusion
        soft_fusion_probs = np.array(list(accuracy_scores.values()))
        soft_fusion_result = np.mean(soft_fusion_probs)

        # Hard fusion
        def hard_fusion_vote(scores):
            threshold = 0.5
            decisions = [1 if score >= threshold else 0 for score in scores]
            final_decision = Counter(decisions).most_common(1)[0][0]
            return final_decision

        hard_fusion_result = hard_fusion_vote(soft_fusion_probs)

        return soft_fusion_result, hard_fusion_result

    # Set up the navigation bar
    st.sidebar.title("Navigation")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    fusion_method = st.sidebar.selectbox("Select Fusion Method", ["Soft Fusion", "Hard Fusion"])

    # If a file is uploaded
    if file is not None:
        data = load_data(file)

        # Preprocessing
        # Define preprocessing steps
        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                'PaymentMethod']
        numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        imputer = SimpleImputer(strategy='mean')

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])

        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(kernel='linear', random_state=42),
            'KNN': KNeighborsClassifier(),
            'River Logistic Regression': RiverLogisticRegression()
        }

        # Sidebar
        st.sidebar.title("Churn Prediction Model")

        # Select classifier
        classifier_name = st.sidebar.selectbox("Select Classifier", list(classifiers.keys()))

        # Main content
        st.title("Churn Prediction Model")

        # Display data
        st.subheader("Data Preview")
        st.write(data.head())

        # Split data into features and target
        X = data.drop(columns=['Churn', 'customerID'])
        y = data['Churn']

        # Preprocess data
        X_preprocessed = preprocessor.fit_transform(X)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

        # Train classifier
        classifier = classifiers[classifier_name]

        if classifier_name == 'River Logistic Regression':
            from river import optim

            # Initialize River Logistic Regression model
            river_logistic_regression = RiverLogisticRegression(optimizer=optim.SGD())

            # Update the model with each instance of data
            for xi, yi in zip(X_train, y_train):
                river_logistic_regression.learn_one(xi, yi)

        else:
            classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy:.2f}")

        if fusion_method == "Soft Fusion":
            soft_fusion_result, _ = fusion({
                "Random Forest": accuracy_score(y_test, classifiers['Random Forest'].predict(X_test)),
                "Decision Tree": accuracy_score(y_test, classifiers['Decision Tree'].predict(X_test)),
                "AdaBoost": accuracy_score(y_test, classifiers['AdaBoost'].predict(X_test)),
                "XGBoost": accuracy_score(y_test, classifiers['XGBoost'].predict(X_test)),
                "Gradient Boosting": accuracy_score(y_test, classifiers['Gradient Boosting'].predict(X_test)),
                "SVM": accuracy_score(y_test, classifiers['SVM'].predict(X_test)),
                "KNN": accuracy_score(y_test, classifiers['KNN'].predict(X_test)),
                "River Logistic Regression": accuracy_score(y_test, [river_logistic_regression.predict_one(xi) for xi in X_test]) if 'River Logistic Regression' in classifiers else accuracy
            })
            st.write("Soft Fusion Result:", soft_fusion_result)
        else:
            _, hard_fusion_result = fusion({
                "Random Forest": accuracy_score(y_test, classifiers['Random Forest'].predict(X_test)),
                "Decision Tree": accuracy_score(y_test, classifiers['Decision Tree'].predict(X_test)),
                "AdaBoost": accuracy_score(y_test, classifiers['AdaBoost'].predict(X_test)),
                "XGBoost": accuracy_score(y_test, classifiers['XGBoost'].predict(X_test)),
                "Gradient Boosting": accuracy_score(y_test, classifiers['Gradient Boosting'].predict(X_test)),
                "SVM": accuracy_score(y_test, classifiers['SVM'].predict(X_test)),
                "KNN": accuracy_score(y_test, classifiers['KNN'].predict(X_test)),
                "River Logistic Regression": accuracy_score(y_test, [river_logistic_regression.predict_one(xi) for xi in X_test]) if 'River Logistic Regression' in classifiers else accuracy
            })
            st.write("Hard Fusion Result:", hard_fusion_result)

    # If no file is uploaded
    else:
        st.write("Please upload a CSV file.")    
except Exception as e:
    print(e)