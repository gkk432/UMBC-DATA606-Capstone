## 1. Title and Author

- **Project Title:** Decoding Consumer Sentiment: Analyzing Amazon Fine Food Reviews with Machine Learning
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Author Name: Gowtham Karaka
- Link to the author's GitHub repo of the project: [Your GitHub Repo Link]
- Link to the author's LinkedIn profile: [Your LinkedIn Profile Link]
- Link to your PowerPoint presentation file: [Your PowerPoint Link]
- Link to your YouTube video: [Your YouTube Video Link]

## 2. Background

- **What is it about?**  
  This project focuses on analyzing customer reviews of fine foods on Amazon to determine sentiment and predict the helpfulness of reviews. It aims to classify reviews as positive or negative based on their text content and evaluate which factors contribute to a review being perceived as helpful.

- **Why does it matter?**  
  Understanding customer sentiment can provide valuable insights for businesses to improve their products and services. Additionally, identifying helpful reviews can guide consumers in making informed purchasing decisions.

- **Research questions:**  
  1. Can we accurately predict the sentiment of a review based on its text?
  2. What features contribute to the perceived helpfulness of reviews?
  3. How do sentiment trends vary over time within the dataset?

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
  - Target variable for helpfulness prediction: Helpfulness ratio (`HelpfulnessNumerator / HelpfulnessDenominator`).

- **Which variables/columns may be selected as features/predictors for your ML models?**  
  - `Text` (review text), `Summary`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, length
