# Email-Spam-Classifier-using-Naive-Bayes
This project is an Email Spam Classifier built using Naive Bayes and TF-IDF vectorization.
It predicts whether an email is Spam or Not Spam based on its text content.

The model is trained on the SMS Spam Collection Dataset from Kaggle.

📂 Dataset

Source: Kaggle – SMS Spam Collection Dataset

Total records: 5572 messages

Columns:

label: "ham" (not spam) or "spam"

message: the content of the email/text

⚙️ Steps in the Project
1️⃣ Data Preprocessing

Removed punctuation and converted text to lowercase

Encoded labels (ham → 0, spam → 1)

Split data into train and test sets

2️⃣ Feature Extraction

Used TF-IDF Vectorizer to convert text into numerical features

Removed common English stopwords

3️⃣ Model Training

Trained a Multinomial Naive Bayes model using the training data

4️⃣ Model Evaluation

Evaluated performance using:

Accuracy

Precision

Recall

F1-score

Visualized results with a Confusion Matrix

📊 Results
Metric	Score
Accuracy	~97.8%
Precision	0.96
Recall	0.92
F1-score	0.94

✅ The model performs very well in distinguishing spam from normal emails.
