## 1. Title and Author

- **Project Title:** Decoding Consumer Sentiment: Analyzing Amazon Fine Food Reviews with Machine Learning
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Author Name: Gowtham Karaka
- Link to the author's GitHub repo of the project: [GitHub Repository](https://github.com/gkk432/UMBC-DATA606-Capstone)
- Link to the author's LinkedIn profile: [LinkedIn Profile](https://www.linkedin.com/in/gowtham-karaka/)
- Link to your PowerPoint presentation file: [PowerPoint Presentation](https://docs.google.com/presentation/d/1e_2jUpg4qzkmpsjad-w8pRmB1xoQRKGx/edit?usp=sharing&ouid=111051529807216348496&rtpof=true&sd=true)
- Link to your YouTube video: [YouTube Video](https://youtu.be/Sc5JFG5M4z4)


## 2. Background

- **What is it about?**  
  This project focuses on analyzing customer reviews of fine foods on Amazon to determine sentiment and predict the helpfulness of reviews. The primary goal is to classify reviews as positive or negative based on their text content and evaluate the factors that contribute to a review being perceived as helpful.

- **Why does it matter?**  
  Understanding customer sentiment provides valuable insights for businesses to enhance their products and services. Identifying helpful reviews also helps consumers make informed purchasing decisions by prioritizing reliable feedback.

- **Research questions:**  
  1. Can we accurately predict the sentiment of a review based on its text?  
  2. Can we determine the confidence level of the sentiment classification?  


## 3. Data

- **Data sources:**  
  Amazon Fine Food Reviews dataset from SNAP and Kaggle.

- **Data size:**  
  Approximately 300 MB.

- **Data shape:**  
  568,454 rows and 10 columns.

- **Time period:**  
  October 1999 to October 2012. 

- **What does each row represent?**  
  Each row represents a single review of a fine food product on Amazon.

- **Data dictionary:**
  - `Id`: Integer - Unique identifier for the review.
  - `ProductId`: String - Unique identifier for the product.
  - `UserId`: String - Unique identifier for the user.
  - `ProfileName`: String - Name of the user.
  - `HelpfulnessNumerator`: Integer - Number of users who found the review helpful.
  - `HelpfulnessDenominator`: Integer - Number of users who indicated whether they found the review helpful or not.
  - `Score`: Integer - Rating between 1 and 5.
  - `Time`: Integer - Timestamp for the review.
  - `Summary`: String - Brief summary of the review.
  - `Text`: String - Text of the review.

- **Which variable/column will be your target/label in your ML model?**  
  - Target variable for sentiment classification: `Score` (converted to binary sentiment: positive if ≥ 4, negative if ≤ 2).
  
- **Which variables/columns may be selected as features/predictors for your ML models?**  
  - `Text` (review text)
  - The primary feature for the sentiment classification model.  
  - Contains the full review written by the customer, providing a rich source of information for natural language processing (NLP).  

## 4. Exploratory Data Analysis (EDA)

- **Focus:**  
  - The primary focus was on the target variable (`Score`) and the feature (`Text`).  
  - All other columns that were not directly contributing to sentiment classification were dropped to simplify the analysis and modeling process.

- **Summary Statistics:**  
  - Produced summary statistics for the key variables:
    - Distribution of `Score` to understand the class imbalance between positive and negative reviews.
    - Analysis of review lengths to identify patterns in the amount of information provided for different sentiments.
    - Distribution of helpfulness metrics (`HelpfulnessNumerator` and `HelpfulnessDenominator`) to evaluate the reliability of review feedback.

- **Visualizations:**  
  - Created various visualizations using Plotly Express and Matplotlib:
    - Bar plots showing the number of reviews per year to understand temporal trends.
    - Word clouds to identify frequent terms in positive and negative reviews.
    - Box plots for review lengths categorized by sentiment to assess variability.
    - Distribution plots for the helpfulness scores to spot anomalies or trends.

- **Data Cleansing:**  
  - **Missing Values:** Checked for missing data in critical columns like `Text` and `Score` to ensure completeness. No significant missing values were found in these columns.  
  - **Duplicate Rows:** Removed duplicate reviews, particularly those repeated across different product flavors, by deduplicating based on `UserId`, `ProfileName`, `Time`, and `Text`.

- **Data Transformation:**  
  - Explored if the dataset needed restructuring:
    - Assessed whether merging or splitting columns was necessary.
    - Ensured the dataset adhered to the "tidy data" principles:  
      - Each row represents one unique review.
      - Each column represents one unique property of the review.
  - Confirmed that additional data sources (e.g., Census data) were not required for this analysis.

- **Textual Data Preprocessing:**  
  - Implemented extensive preprocessing steps to clean and prepare the `Text` column for machine learning models:
    - **Normalization:** Converted text to lowercase and removed punctuation for uniformity.  
    - **Contraction Expansion:** Expanded contractions (e.g., "isn't" → "is not") using a predefined mapping.
    - **Tokenization:** Split text into individual words for further processing.
    - **Stopword Removal:** Removed common but uninformative words (e.g., "the," "and") using NLTK's stopword list.
    - **Lemmatization:** Reduced words to their base forms (e.g., "running" → "run") using NLTK's WordNetLemmatizer.

- **Outcome:**  
  - After cleaning and preprocessing, the dataset was ready for machine learning, with a tidy structure and meaningful text features for analysis and modeling.

## 5. Model Training

- **Models Used for Predictive Analytics:**  
  - **Multinomial Naive Bayes (TF-IDF):** Selected for its simplicity and efficiency in handling text classification tasks.  
  - **Multinomial Naive Bayes with SMOTE:** Implemented to address class imbalance by oversampling the minority class.  
  - **Logistic Regression (TF-IDF):** Chosen for its balanced performance, interpretability, and robustness in binary classification.  
  - **Logistic Regression with SMOTE:** Used to improve the recall for minority class predictions while maintaining overall accuracy.  
  - **Random Forest (TF-IDF):** Leveraged for its ensemble approach, handling complex patterns in the data effectively.  

- **Training Strategy:**  
  - The dataset was split into **80% training** and **20% testing** using `train_test_split` to ensure an unbiased evaluation of the models.  
  - **TF-IDF Vectorization** was applied to the `Text` column to convert textual data into numerical features suitable for machine learning.  
  - For models requiring imbalance handling, **SMOTE** (Synthetic Minority Oversampling Technique) was applied to oversample the minority class during training.

- **Python Packages Used:**  
  - **scikit-learn:** For implementing machine learning models, splitting data, vectorization, and performance evaluation.  
  - **imbalanced-learn:** For applying SMOTE to balance the dataset for specific models.  
  - **NLTK:** For preprocessing text data, including tokenization, stopword removal, and lemmatization.  

- **Development Environment:**  
  - **Google Colab:** Used for model development and leveraging GPU resources for efficient processing.  
  - **Streamlit Cloud:** Deployed trained models via an interactive web app for real-time sentiment analysis.  
  - **GitHub:** Utilized for project management, version control, and collaboration.  

- **Performance Measurement and Comparison:**  
  - Models were evaluated and compared using the following metrics:
    - **Accuracy:** The percentage of correctly classified reviews.  
    - **Precision:** How well the model predicts positive instances without false positives.  
    - **Recall:** The ability of the model to identify all actual positive reviews (sensitivity).  
    - **F1-Score:** A balance between precision and recall for better evaluation of imbalanced classes.  
    - **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.  
  - Performance across all models (with and without SMOTE) was analyzed to determine the best-performing approach for sentiment classification.  

## 6. Web Application for Sentiment Analysis

- **Tool Used:**  
  - Developed using **Streamlit**, a Python-based framework for creating interactive web applications.

- **Purpose of the App:**  
  - To allow users to interact with the trained sentiment analysis model.  
  - Users can input a review, and the app predicts whether the sentiment is **Positive** or **Negative**.  

- **Features of the App:**  
  - **Model Integration:** The app integrates the trained **Logistic Regression model with TF-IDF** for sentiment classification.  
  - **Input Box:** Users can input custom reviews directly in the app for analysis.  
  - **Prediction Results:** The app displays:
    - Predicted sentiment (Positive or Negative).  
    - Confidence score for the prediction.  

- **Key Functionalities:**  
  - **Real-Time Analysis:** The app processes user input in real time and generates predictions.  
  - **Ease of Access:** Simple and user-friendly interface, accessible via a web browser.  

- **Deployment Platform:**  
  - The app is hosted on **Streamlit Cloud**, enabling easy access and scalability.  

- **Advantages of Using Streamlit:**  
  - Rapid development and deployment of interactive apps with minimal code.  
  - Lightweight and requires no advanced web development skills.  

- **How to Access the App:**  
  - The app is available at "https://umbc-data606-capstone-pccvrpjnfhwqh2stgbgrwu.streamlit.app/"  
  - Users can try out their own reviews or explore the model's sentiment predictions.

## 7. Conclusion

## Summary of Work and Potential Application
- Developed a sentiment analysis model using the **Amazon Fine Food Reviews** dataset, leveraging techniques like **TF-IDF vectorization** and **Logistic Regression** for high accuracy (92.51%).  
- Built an interactive **Streamlit web app** to allow users to input reviews and receive real-time sentiment predictions.  
- The app can be applied to:
  - **E-commerce platforms** for analyzing customer reviews and identifying trends.  
  - **Businesses** to understand customer feedback and improve product offerings.  
  - **Market Research** to gauge customer sentiment towards products or brands.

## Limitations
- **Class Imbalance:** Negative reviews were underrepresented, which slightly impacted the model's recall for the minority class.  
- **Context Limitations:** The model does not consider advanced contextual nuances or sarcasm that may affect sentiment accuracy.  
- **Single Language:** The model only works with English text and would require adaptation for multilingual datasets.  

## Lessons Learned
- **Data Preprocessing:** Importance of thorough text preprocessing (e.g., stopword removal, lemmatization) for effective machine learning.  
- **Class Imbalance Handling:** Learned to use techniques like SMOTE to address imbalanced datasets, though it can sometimes lead to reduced precision.  
- **Model Comparison:** Gained insights into evaluating models based on multiple metrics (e.g., precision, recall, F1-score) to select the best-performing one.  
- **Web App Deployment:** Mastered deploying machine learning models to a live environment using Streamlit Cloud.

## 8. Future Research Directions
- **Multilingual Sentiment Analysis:** Expand the model to handle reviews in multiple languages using advanced NLP techniques (e.g., translation APIs or multilingual embeddings).  
- **Context-Aware Models:** Explore transformer-based models like **BERT** or **GPT** to better understand context, sarcasm, and nuanced language.  
- **Incorporate Metadata:** Include additional features like product categories, user demographics, or review helpfulness ratings to enhance predictions.  
- **Dynamic Web App:** Improve the app to handle batch uploads of reviews for bulk sentiment analysis and provide detailed visualizations for users.  


