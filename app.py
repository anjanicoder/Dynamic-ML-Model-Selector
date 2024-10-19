# Required libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Default dataset path
DEFAULT_DATASET_PATH = 'heart.csv'

# Function to load the default dataset
def load_default_data():
    return pd.read_csv(DEFAULT_DATASET_PATH)

# Streamlit app title
st.title('Dynamic ML Model Selector (Classifier/Regressor)')

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

# Load the uploaded dataset or default dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview (Uploaded):")
    st.dataframe(df.head())
else:
    df = load_default_data()
    st.write("Dataset Preview (Default):")
    st.dataframe(df.head())

# Step 2: Check for null values and handle them
null_values = df.isnull().sum().sum()
if null_values > 0:
    st.warning(f"Your dataset contains {null_values} null values. These will be dropped.")
    df = df.dropna()  # Drop rows with any null values
    st.write("Dataset after dropping null values:")
    st.dataframe(df.head())

# Step 3: Select input features and target variable
target = st.selectbox("Select target variable", df.columns)
all_features = df.columns.difference([target]).tolist()

# Checkbox to select all features
select_all = st.checkbox("Select all features", value=False)
if select_all:
    features = all_features
else:
    features = st.multiselect("Select input features (Hold 'Ctrl' or 'Command' to select multiple)", all_features)

# Ensure features are selected
if len(features) == 0:
    st.warning("Please select at least one input feature.")
else:
    # Step 4: Encode categorical features if present
    le = LabelEncoder()
    col = df.select_dtypes('object')

    for c in col.columns:
        df[c] = le.fit_transform(df[c])

    # Split data into features (X) and target (y)
    X = df[features].values
    y = df[target].values

    # Step 5: User specifies if target is categorical or numerical
    target_type = st.radio("Is the target variable categorical or numerical?", ('Categorical', 'Numerical'))

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Standardize the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Step 8: Dynamically choose model based on target type
    label_classes = None
    predictions = None
    model = None
    accuracy, mse, r2 = None, None, None

    if target_type == 'Numerical':
        model_option = st.selectbox("Select Model (Regression)", ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor'])

        if model_option == 'Linear Regression':
            model = LinearRegression()
        elif model_option == 'Decision Tree Regressor':
            model = DecisionTreeRegressor()
        elif model_option == 'Random Forest Regressor':
            random_grid = {
                'n_estimators': [10, 20, 30, 50, 100],
                'max_depth': [6, 7, 8, 9, 10],
                'min_samples_split': [5, 10, 20]
            }
            rf = RandomForestRegressor(random_state=42)
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                           n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
            rf_random.fit(X_train, y_train)
            model = rf_random.best_estimator_
            st.write(f"Best Parameters: {rf_random.best_params_}")

    elif target_type == 'Categorical':
        model_option = st.selectbox("Select Model (Classification)", ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier'])
        
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        label_classes = le.classes_

        if model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Decision Tree Classifier':
            model = DecisionTreeClassifier(criterion='entropy')
        elif model_option == 'Random Forest Classifier':
            random_grid = {
                'n_estimators': [10, 20, 30, 50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [6, 7, 8, 9, 10],
                'min_samples_split': [5, 10, 20]
            }
            rf = RandomForestClassifier(random_state=42)
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                           n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
            rf_random.fit(X_train, y_train)
            model = rf_random.best_estimator_
            st.write(f"Best Parameters: {rf_random.best_params_}")

    # Step 9: Train the selected model
    if model:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if target_type == 'Numerical':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R2 Score: {r2}")

        elif target_type == 'Categorical':
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy}")

    # Step 10: Visualization
    if accuracy is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,  # converting to percentage
            title={'text': "Model Accuracy"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}]
            }
        ))
        st.plotly_chart(fig)

    if mse is not None:
        fig = go.Figure([go.Bar(x=["Mean Squared Error"], y=[mse], marker_color='indianred')])
        fig.update_layout(title="Mean Squared Error", yaxis=dict(title="MSE"))
        st.plotly_chart(fig)

    # Confusion Matrix for classification models
    if predictions is not None and model_option in ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier']:
        st.write("Confusion Matrix Visualization")
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        st.pyplot(plt)



