import streamlit as st
import google.generativeai as genai

# Set up the Gemini API key
api_key = "AIzaSyDlaUJZyR8zsIFBZFRI5mJbIOk4ojxBTmA"
genai.configure(api_key=api_key)

# Streamlit Title
st.title("Learn, Quiz, and Master: Machine Learning Algorithms 🧠🔍🤖")
st.subheader('Explore Models and Understand Algorithms 📊')

# Sidebar for Model Selection
with st.sidebar:
    linkedin_url = "https://www.linkedin.com/in/inisha-r/"
    github_url = "https://github.com/inisha-r"

    # You can use online icon URLs, or upload your own icons and use them.
    linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"  # LinkedIn icon URL
    github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"  # GitHub icon URL

    st.markdown(f"""
        Project By:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color='black'>**Inisha Sallove R**</font><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src='{linkedin_icon}' alt='LinkedIn' width='20'/> [LinkedIn]({linkedin_url}) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src='{github_icon}' alt='GitHub' width='20'/> [GitHub]({github_url}) <br>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("Select Model")

    # Step 1: Select analysis type
    analysis_type = st.selectbox("Select Analysis Type", [
        "Stock Market Prediction", "Binary Classification","Multi-Class Classification","Time Series Forecasting","Clustering","Anomaly Detection","Natural Language Processing","Recommendation Systems","Sentiment Analysis","Image Classification","Object Detection","Dimensionality Reduction"
    ])

    # Step 2: Model selection based on the analysis type
    if analysis_type == "Stock Market Prediction":
        model_choice = st.selectbox("Choose a Model", [
            "Linear Regression","Decision Tree Regressor","K-Nearest Neighbors (KNN) Regressor","Random Forest Regressor","Support Vector Regressor (SVR)"
        ])
    elif analysis_type == "Binary Classification":
        model_choice = st.selectbox("Choose a Model", [
            "Logistic Regression","Decision Tree","K-Nearest Neighbors","Random Forest Classifier","Support Vector Machine (SVM)","Naive Bayes"
        ])
    elif analysis_type == "Multi-Class Classification":
        model_choice = st.selectbox("Choose a Model", [
            "Decision Tree","Random Forest","K-Nearest Neighbors","Logistic Regression (with one-vs-rest)","Naive Bayes"
        ])
    elif analysis_type == "Time Series Forecasting":
        model_choice = st.selectbox("Choose a Model", [
            "Linear Regression","Decision Tree Regressor","Random Forest Regressor","ARIMA","Prophet"
        ])
    elif analysis_type == "Clustering":
        model_choice = st.selectbox("Choose a Model", [
            "K-Means","DBSCAN","Agglomerative Clustering","Gaussian Mixture Model","Hierarchical Clustering"
        ])
    elif analysis_type == "Anomaly Detection":
        model_choice = st.selectbox("Choose a Model", [
            "Isolation Forest","One-Class SVM","Local Outlier Factor (LOF)"
        ])
    elif analysis_type == "Natural Language Processing":
        model_choice = st.selectbox("Choose a Model", [
            "TF-IDF + Logistic Regression","Naive Bayes (Text Classification)","Support Vector Machine (SVM)","LSTM (for Text Classification)"
        ])
    elif analysis_type == "Recommendation Systems":
        model_choice = st.selectbox("Choose a Model", [
            "Collaborative Filtering","Content-Based Filtering","Matrix Factorization (SVD)"
        ])
    elif analysis_type == "Sentiment Analysis":
        model_choice = st.selectbox("Choose a Model", [
            "Naive Bayes (Sentiment Analysis)","Logistic Regression (Sentiment Analysis)","Support Vector Machine (SVM)","LSTM (Sentiment Analysis)"
        ])
    elif analysis_type == "Image Classification":
        model_choice = st.selectbox("Choose a Model", [
            "Convolutional Neural Networks (CNN)","ResNet","VGGNet","Inception","MobileNet"
        ])
    elif analysis_type == "Object Detection":
        model_choice = st.selectbox("Choose a Model", [
            "YOLO (You Only Look Once)","Faster R-CNN","SSD (Single Shot Multibox Detector)","RetinaNet"
        ])
    elif analysis_type == "Dimensionality Reduction":
        model_choice = st.selectbox("Choose a Model", [
            "PCA (Principal Component Analysis)","LDA (Linear Discriminant Analysis)","t-SNE","UMAP (Uniform Manifold Approximation and Projection)"
        ])

    # Step 3: Run model button
    run_button = st.button("Run Model")

# Function to fetch model details from Gemini API
def fetch_model_details(model_choice):
    try:
        prompt = f"Provide a detailed explanation of the {model_choice} algorithm in simple terms, as if explaining to a student. Cover the core concepts, how the algorithm works step by step, its strengths and limitations, and include real-world examples of where it's commonly used."
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch model details: {e}")
        return f"Could not retrieve details for {model_choice}."

# Function to add example for the selected algorithm
def add_examples(model_choice):
    examples = {
        "Linear Regression": "Example: Predicting house prices based on square footage.","Logistic Regression": "Example: Predicting whether a student will pass or fail based on hours studied.","Decision Tree": "Example: Classifying if a customer will churn based on their usage data.","K-Means": "Example: Clustering customers based on their purchasing habits.","Random Forest Classifier": "Example: Predicting if a loan application will be approved based on applicant data.","Random Forest Regressor": "Example: Predicting sales revenue based on advertising spend and seasonality.","K-Nearest Neighbors": "Example: Classifying if an email is spam based on previous email data.","Support Vector Machine (SVM)": "Example: Classifying handwritten digits based on pixel intensity features.","Naive Bayes": "Example: Predicting the sentiment of a movie review (positive/negative).","PCA (Principal Component Analysis)": "Example: Reducing dimensionality of gene expression data for visualization.","LDA (Linear Discriminant Analysis)": "Example: Classifying species of flowers based on petal and sepal measurements.","LSTM (Text Classification)": "Example: Classifying email content into categories like 'work', 'personal', or 'spam'.","Collaborative Filtering": "Example: Recommending movies to users based on similar users' preferences.","Content-Based Filtering": "Example: Recommending products based on features of previously purchased items.","ARIMA": "Example: Forecasting the stock price of a company based on historical price trends.","Prophet": "Example: Forecasting website traffic over time with daily and seasonal patterns.","Isolation Forest": "Example: Detecting fraudulent transactions in a set of financial data.","YOLO (You Only Look Once)": "Example: Detecting cars, pedestrians, and traffic signs in real-time video streams.","Faster R-CNN": "Example: Detecting multiple objects in an image, such as humans and animals.","t-SNE": "Example: Visualizing high-dimensional image data in a 2D scatter plot for clustering analysis.","UMAP": "Example: Visualizing customer segmentation based on shopping behavior data.","Matrix Factorization (SVD)": "Example: Recommending books to users by factoring user preferences and item similarities."
    }

    if model_choice in examples:
        st.write(f"Example Usage: {examples[model_choice]}")

# Function to fetch quiz questions from Gemini API
def fetch_quiz(model_choice):
    try:
        prompt = f"Create 5 simple quiz questions with 4 options each about the {model_choice} algorithm to test the user's knowledge."
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch quiz: {e}")
        return "Could not retrieve quiz."

# Function to display quiz examples without requiring user input
def display_quiz(model_choice):
    quiz_questions = fetch_quiz(model_choice)
    st.subheader("Test Your Knowledge!")
    st.write(quiz_questions)

# Main workflow
if run_button:
    # Fetch details about the selected model using Gemini API
    model_info = fetch_model_details(model_choice)
    st.write(model_info)

    # Add examples for the selected model
    add_examples(model_choice)

    # Display example quiz questions for the selected model
    display_quiz(model_choice)
