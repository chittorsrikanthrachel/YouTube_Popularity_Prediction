# 🎥 YouTube Popularity Prediction – A Machine Learning Driven Approach

A web-based machine learning application that predicts the popularity of YouTube videos by estimating video views and likes, recommending relevant hashtags, and providing analytical insights to help content creators make data-driven decisions.

---

## 🚀 Project Overview

With the increasing competition among YouTube creators, predicting video performance before publishing has become increasingly important. This project combines machine learning models, analytics, and a web interface to help creators estimate video performance and optimize their content strategy.

The application provides:

- 📈 Video Views Prediction
- ❤️ Video Likes Prediction
- 🏷️ Hashtag Recommendation
- 📊 Insights & Visualizations
- 🤖 Interactive Chatbot Interface

The primary goal is to assist YouTube creators in improving video visibility, audience engagement, and overall channel performance using machine learning and data analytics.

---

## ✨ Features

### 📈 Video Views Prediction

Predicts the expected number of views using multiple regression models based on:

- Video metadata
- Historical engagement
- Channel statistics

Available prediction models:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Users can choose models through user-friendly labels based on accuracy, speed, and stability.

---

### ❤️ Video Likes Prediction

Predicts the expected number of likes using a Random Forest Regressor trained on historical engagement data.

Input features include:

- Recent views
- Recent dislikes
- Recent comments
- Video category
- Number of tags
- Description length
- Title length
- Publish day

---

### 🏷️ Hashtag Recommendation

Suggests relevant hashtags using a dictionary-based recommendation approach.

Recommendations are generated using:

- Video title
- Video category

The goal is to improve content discoverability on YouTube.

---

### 📊 Insights & Analytics

After predicting video views, users can access analytical dashboards including:

- Best upload time heatmap
- Average views by upload hour
- Average views by upload day
- Model comparison
- Engagement breakdown

These insights help creators optimize publishing strategies.

---

### 🤖 Interactive Chatbot

The application includes a chatbot that supports:

- Voice interaction
- Text interaction
- Module navigation
- User guidance

The chatbot helps users navigate the application and understand available features.

---

## 🛠️ Technologies Used

### 💻 Programming Language

- Python

### 🤖 Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting Machine

### ⚙️ Backend

- Flask

### 🎨 Frontend

- HTML
- CSS
- JavaScript

### 📂 Data Processing

- Pandas
- NumPy

### 📉 Data Visualization

- Matplotlib
- Seaborn

### 🧰 Development Tools

- Jupyter Notebook
- Python IDLE

---

## 📊 Dataset

The project uses two datasets collected using the YouTube Data API.

### 📈 Views Dataset

Includes features such as:

- Video duration
- Category
- Publish day/month
- Trending day/month
- Subscriber count
- Video count
- Past likes
- Past dislikes
- Past comments

**Target Variable**

- Video Views

### ❤️ Likes Dataset

Includes features such as:

- Recent 1 hour views
- Recent 1 hour likes
- Recent 1 hour dislikes
- Video category
- Number of tags
- Description length
- Title length
- Publish day

**Target Variable**

- Video Likes

Each dataset contains approximately 5,000 records and is managed using an SQL database.

---

## 🧠 Machine Learning Models

### Views Prediction

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### Likes Prediction

- Random Forest Regressor

---

## 🔮 Future Scope

Possible future improvements include:

- Deep learning-based prediction models
- Dynamic hashtag recommendation
- Sentiment analysis
- Thumbnail image analysis
- Cloud deployment
- Enhanced chatbot capabilities
- Real-time analytics

---

## ⚠️ Limitations

- Limited dataset size
- Designed primarily for channels with 100K–500K subscribers
- Viral videos may affect prediction accuracy
- Dictionary-based hashtag recommendation
- Limited feature engineering

---

## 📌 Conclusion

This project demonstrates how machine learning and data analytics can be combined to assist YouTube creators in predicting video performance, generating meaningful insights, recommending hashtags, and supporting data-driven content strategies through an integrated web application.

---

## 👩‍💻 Author

**C.S.Rachel**

*M.Sc. Data Science*
