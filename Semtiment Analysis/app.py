import streamlit as st
import pandas as pd
import re
import pickle
from io import BytesIO
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Ensure NLTK stopwords are downloaded
def download_nltk_data():
    nltk.download('stopwords')

# Set up stopwords and model path
download_nltk_data()
STOPWORDS = set(stopwords.words("english"))
MODEL_PATH = "C:/Users/Dell/Desktop/ss/Model/"

# Load the models
def load_model(file_name):
    with open(MODEL_PATH + file_name, "rb") as f:
        return pickle.load(f)

# Cache data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Single prediction function
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    return sentiment_mapping(y_predictions)

# Bulk prediction function
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))
    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    graph = get_distribution_graph(data)
    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("#40A578", "#03AED2", "#577B8D", "#3572EF", "#3AA6B9")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()

    # Calculate explode values dynamically
    explode = [0.01] * len(tags)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,  # Use dynamically calculated explode values
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph


# Sentiment mapping
def sentiment_mapping(x):
    if x == 0:
        return "Positive"
    elif x == 1:
        return "Strong Positive"
    elif x == 2:
        return "Neutral"
    elif x == 3:
        return "Negative"
    elif x == 4:
        return "Strong Negative"

# Main function to define the app structure
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "About Data and Model", "Prediction", "About Us"])

    # Upper border
    def upper_border():
        st.markdown(
            """
            <div style="background-color: #2596be; padding: 10px; border-radius: 5px;">
                <h1 style="color: white; text-align: center;">Sentiment Analysis App</h1>
            </div>
            """, unsafe_allow_html=True
        )

    # Home page
    if page == "Home":
        upper_border()
        st.write("            ")
        st.write("Sentiment analysis is gaining popularity in the research community. It assigns positive or negative polarity to entities or items using various natural language processing tools and predicts the performance of different sentiment classifiers.")
        st.write("Our work focuses on sentiment analysis derived from product reviews, employing original text search techniques. These reviews can be classified as positive or negative based on specific aspects related to a query. Additionally, our focus is on multi-label sentiment analysis, where multiple sentiments can be assigned to a single review.")
        st.write("""This app uses a sentiment analysis model trained on a dataset of text samples.
            The model can predict the sentiment of text as Positive, Strong Positive, Neutral,
            Negative, or Strong Negative.""")
    # About Data and Model page
    elif page == "About Data and Model":
        upper_border()
        st.title("About the Data and Model")
        st.write("""
            The data used for training the sentiment analysis model was scraped from Amazon reviews.
        This dataset consists of text samples from various product reviews, covering a wide range
        of products and categories. Then I labeled the sentiment categories such as
        Positive, Strong Positive, Neutral, Negative, or Strong Negative.

        The sentiment analysis model was built using the Random Forest Classifier algorithm.
        This algorithm is effective for multiclass classification tasks and has been trained
        on the scraped Amazon review data to predict the sentiment of text samples.

        By leveraging machine learning techniques, such as Random Forest Classifier, this app
        is capable of analyzing text data from product reviews and predicting sentiment labels
        to provide valuable insights for decision-making.
        """)

    # Prediction page
    elif page == "Prediction":
        upper_border()
        st.title("Prediction")

        # Load models
        predictor = load_model("model_RF.pkl")
        scaler = load_model("scaler.pkl")
        cv = load_model("countVectorizer.pkl")

        # File upload section
        st.subheader("Upload CSV file for bulk prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            st.success("Predictions generated successfully!")
            st.download_button(
                label="Download Predictions",
                data=predictions,
                file_name="Predictions.csv",
                mime="text/csv"
            )
            st.image(graph.getvalue(), caption="Sentiment Distribution", use_column_width=True)

        # Text input section
        st.subheader("Enter text for single prediction")
        text_input = st.text_area("Enter text here")
        if st.button("Predict Sentiment"):
            if text_input:
                predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
                st.success(f"Predicted Sentiment: {predicted_sentiment}")
            else:
                st.error("Please enter text for prediction")

    
    # About Us page
    elif page == "About Us":
        # Upper border
        st.markdown(
            """
            <div style="background-color: #2596be; padding: 10px; border-radius: 5px;">
                <h1 style="color: white; text-align: center;">Sentiment Analysis App</h1>
            </div>
            """, unsafe_allow_html=True
        )
        st.title("About Us")
        # First team member with image on left and text on right
        st.markdown("<h3 style='color: #2596be;'>Musharaf Ahsan</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("C:/Users/Dell/Desktop/ss/111.jpg", use_column_width=True, output_format='JPEG', width=None)
        with col2:
            st.write("""
                A Google Certified Data Analyst and a passionate BSc Computer Science student with a keen interest in the fascinating world of data science. 
                I'm on a mission to harness the power of data to drive insightful decisions and create meaningful impact.
                Although I am yet to acquire professional experience in the field. I've also honed my soft skills, such as teamwork and communication, through collaborative projects and group discussions.
            """)

        # Second team member with image on right and text on left
        st.markdown("<h3 style='color: #2596be;'>Muhammad Akif</h3>", unsafe_allow_html=True)
        col3, col4 = st.columns([2, 1])
        with col3:
            st.write("""
                A BSc Computer Science student deeply passionate about data science. Working as a freelance Lead Generation Manager on Upwork and also a YouTube content creator. My aims to leverage data to drive insightful decisions and create meaningful impact.
            """)
        with col4:
            st.image("C:/Users/Dell/Desktop/ss/222.jpg", use_column_width=True, output_format='JPEG', width=None)


if __name__ == "__main__":
    main()

