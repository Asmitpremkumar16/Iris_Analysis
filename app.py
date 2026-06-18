# Import modules and libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Set the UI

st.set_page_config(page_title="Iris EDA & ML App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose an Option", ["Dataset Overview", "EDA", "ML Model", "Prediction"])

# Load dataset 

data= pd.read_csv(r'iris.csv')

# Overview

if page == "Dataset Overview":
    st.title("Iris Dataset Overview")

    # Info------->

    st.info("This application performs Exploratory Data Analysis (EDA) on the Iris dataset to uncover insights "
            "about feature distributions and relationships. "
            "These insights are used to build a machine learning model that predicts the species of an Iris flower.")



    st.subheader("Sample Data")
    st.dataframe(data.sample(5))

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

    st.subheader("Missing Values")
    st.write(data.isna().sum())

#  EDA

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    # Plot Engine 

    plot_engine = st.sidebar.selectbox(
    "Visualization Engine",
    ["Seaborn (EDA)", "Plotly (Visualization)"]
    )

    st.subheader("Distribution Plot")
    
    # First plot_engine(Seaborn)

    if plot_engine == "Seaborn (EDA)":

        feature = st.selectbox("Select a feature", data.drop(columns=['type']).columns)


        # 1.Histogram

        st.subheader("1. Histogram")

        fig, ax = plt.subplots(figsize= (5,3))
        sns.histplot(data, x=feature, hue='type', kde=True, ax=ax)

        ax.set_title(f"Distribution of {feature} by Species", fontsize=11)
        ax.set_xlabel(feature.replace("_", " ").title(), fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()


        st.pyplot(fig)



        # Info---------->


        feature_insights = {
        "sepal_length": 
            "Sepal length changes across species, but there is some overlap between them. "
            "This means it helps with classification, but it is not enough on its own.",

        "sepal_width": 
            "Sepal width looks quite similar for most species, with a lot of overlap. "
            "Because of this, it is not a very strong feature by itself.",

        "petal_length": 
            "Petal length shows clear differences between species, with very little overlap. "
            "This makes it one of the most useful features for identifying the species.",

        "petal_width": 
            "Petal width separates the species very clearly. "
            "It plays a major role in predicting the correct Iris species."
    }

        st.info(feature_insights.get(feature, "Select a feature to see what the plot tells us."))


        # 2.Boxplot

        st.subheader("2. Boxplot")

        
        fig, ax = plt.subplots(figsize=(5,3))  # reduced size

        sns.boxplot(data=data, x="type", y=feature,hue= 'type', ax=ax)

        ax.set_title(f"Box Plot of {feature} by Species", fontsize=11)
        ax.set_xlabel("Species", fontsize=9)
        ax.set_ylabel(feature.replace("_", " ").title(), fontsize=9)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

        st.pyplot(fig)

        # Info------>

        st.info(
        "The box plot highlights the median, spread, and possible outliers of the selected feature. "
        "Notice how some species have tighter ranges while others are more spread out. "
        "Distinct median positions indicate stronger feature discrimination."
        )


        # 3. Countplot

        st.subheader("3. Countplot")

        fig, ax= plt.subplots(figsize= (5,3))
        sns.countplot(data= data.dropna().drop_duplicates(), x= 'type', hue="type", ax= ax)

        ax.set_title("Class Distribution of Iris Species", fontsize=11)
        ax.set_xlabel("Species", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        df = pd.DataFrame({
        "Species": ["Setosa", "Versicolor", "Virginica"],
        "Count": [48, 50, 49]
        })

        st.dataframe(df)

        st.info("After preprocessing, the dataset shows a nearly balanced distribution of Iris species. "
                "Setosa has 48 samples, Versicolor has 50 samples, and Virginica has 49 samples.")
    

    # Second plot_engine(Interactive)

    elif plot_engine == "Plotly (Visualization)":

        st.subheader("1. Scatterplot")

        # 1.Scatterplot

        x_feature = st.selectbox(
        "Select X-axis feature",
        [col for col in data.columns if col != "type"]
        )

        y_feature = st.selectbox(
        "Select Y-axis feature",
        [col for col in data.columns if col != "type"],
        index=1
        )

        fig = px.scatter(
        data,
        x=x_feature,
        y=y_feature,
        color="type",
        title=f"{x_feature} vs {y_feature}"
        )

        st.plotly_chart(fig, use_container_width=True)


        # Info----------->
        st.info("This scatter plot compares two features to highlight patterns and class separation among species. "
            "Clear clustering suggests that the selected feature pair is effective for classification."
        )



        # 2.Heatmap

        st.subheader("2. Heatmap")

        corr = data.drop("type", axis=1).corr()

        fig = px.imshow(
        corr,
        text_auto=True,
        title="Feature Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)


        # Info--------->
        st.info("The correlation heatmap highlights relationships between features. "
                "Highly correlated features may carry similar information, which is useful when selecting features for modeling."
        )

        # 3.Species-wise Mean Analysis


        st.subheader("3. Species-wise Mean Values")

        summary_df = (data.groupby("type") .mean() .round(2) .reset_index() .rename(columns={"type": "Species"}))

        st.dataframe(summary_df, use_container_width=True)


        # Info-------->

        st.info("The species-wise mean values highlight how petal features differ significantly across species, "
                "while sepal features show more overlap. These differences help guide feature selection."
            )

# ML Model

elif page == "ML Model":
    st.title("Machine Learning Model")

    X = data.drop('type', axis=1)
    y = data['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=5)

    pipeline= Pipeline([
        ("Scaler", StandardScaler()),
        ("Model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred= pipeline.predict(X_test)

    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    # Info for Model Performance--------->

    st.info("Model performance metrics evaluate how well the classifier predicts each species. "
            "Precision shows how accurate the model's positive predictions are, recall indicates how well "
            "the model identifies all actual samples of a species, and the F1-score balances both precision "
            "and recall. Values closer to 1 indicate better performance, while lower values highlight areas "
            "where the model may be misclassifying samples."
        )


    st.subheader("Confusion Matrix")
    st.dataframe(confusion_matrix(y_test, y_pred))

    # Info for Confusion Matrix------------->

    st.info("The confusion matrix provides a detailed breakdown of the model’s predictions. "
            "Each row represents the actual species, while each column represents the predicted species. "
            "Diagonal values show correct predictions, and off-diagonal values highlight misclassifications. "
            "A stronger model will have higher values along the diagonal."
        )


    report_dict = classification_report(y_test, y_pred, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose().round(2)
    report_df.rename(columns={
        "precision": "Precision (0–1)",
        "recall": "Recall (0–1)",
        "f1-score": "F1 Score (0–1)"
    }, inplace=True)

    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)

    # Info for Classification Report----------->

    st.info("Precision, recall, and F1-score range from 0 to 1, where 1 means perfect prediction. "
            "The support column shows how many actual samples of each species were present in the test dataset. "
            "Support is a count, not a performance metric."
        )

# 8.Prediction------------------------------------>

elif page == "Prediction":
    st.title("Predict Iris Species")

    sl = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sw = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
    pl = st.number_input("Petal Length", 1.0, 7.0, 1.4)
    pw = st.number_input("Petal Width", 0.1, 2.5, 0.2)

    X = data.drop('type', axis=1)
    y = data['type']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X, y)

    if st.button("Predict"):
        prediction = model.predict([[sl, sw, pl, pw]])[0]
        st.success(f"Predicted Species: {prediction}")



