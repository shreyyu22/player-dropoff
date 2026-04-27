#  Player Performance Analytics & Difficulty Recommendation System

##  Live Application
https://shreyyu22-player-dropoff-app-2snojk.streamlit.app/

---

##  Overview

This project is an end-to-end **Data Analytics + SQL + Machine Learning + Streamlit** solution designed to analyze player performance in an online FPS game and generate **dynamic difficulty recommendations**.

Players are classified into:
- **Struggling**
- **Balanced**
- **Overperforming**

Based on this classification, the system recommends:
- Decrease Difficulty  
- Keep Difficulty Same  
- Increase Difficulty  

The goal is to improve **player engagement, retention, and gameplay balance**.

---

##  Problem Statement

Online games often lose players when the difficulty is not aligned with player skill.

- Too difficult → Players quit early  
- Too easy → Players get bored  

This project uses gameplay data to provide **data-driven difficulty recommendations**.

---

##  Project Objective

- Analyze player performance using SQL and Python  
- Segment players based on behavior and skill  
- Build ML models to classify players  
- Generate difficulty recommendations  
- Deploy an interactive web application  

---

##  Tech Stack

| Category | Tools |
|--------|------|
| Data Analysis | Python, Pandas, NumPy |
| SQL Analysis | Advanced SQL Queries |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Models | K-Means, Logistic Regression, Random Forest |
| Deployment | Streamlit |
| Storage | Joblib |
| Version Control | GitHub |

---

##  Dataset

- Source: Kaggle (Call of Duty Player Dataset)  
- File: `cod.csv`  

### Key Features:
- kdRatio  
- scorePerMinute  
- wins, losses  
- timePlayed  
- gamesPlayed  
- hits, shots  
- xp, level, prestige  


## 🏗️ Project Architecture

Dataset  
→ SQL Analysis Layer  
→ Exploratory Data Analysis (EDA)  
→ Feature Engineering  
→ K-Means Clustering  
→ Logistic Regression + Random Forest  
→ Majority Voting  
→ Difficulty Recommendation  
→ Streamlit Application

---

## 🧮 SQL Analysis

SQL is used as a **core analytical layer** to extract insights from player data.

### 🔍 Key SQL Insights

### 1. Performance Ranking
- Top players have significantly higher performance scores  
- Quartile segmentation separates elite players from average players  
- High-ranking players require increased difficulty  

---

### 2. Engagement vs Performance
- Low engagement players are mostly struggling  
- High engagement players perform more consistently  
- Low engagement players are at higher drop-off risk  

---

### 3. Win Rate Analysis
- Win rate shows player consistency  
- Combining win rate and score per minute improves performance evaluation  
- Helps identify consistently strong players  

---

### 4. Accuracy Analysis
- Accuracy strongly correlates with player skill  
- High accuracy players perform better overall  
- Low accuracy players may need easier difficulty  

---

### 5. Drop-Off Risk Detection
- Low playtime + low K/D ratio → high drop-off risk  
- Early intervention can improve player retention  
- Identifies players likely to quit early  

---

### 6. Recommendation Validation
- Struggling → Decrease Difficulty  
- Balanced → Keep Difficulty Same  
- Overperforming → Increase Difficulty  
- Recommendations align with player behavior  

---

### 7. Outlier Detection
- High outliers represent expert players  
- Low outliers represent inactive or struggling players  
- Helps detect extreme performance patterns  

---

### 8. Player Scoring Index
- Multi-feature scoring gives better ranking than single metrics  
- Players can be classified into Beginner, Casual, Competitive, Elite  
- Useful for matchmaking and difficulty balancing

## 🤖 Machine Learning Approach

### K-Means Clustering
Segments players into:
- Struggling  
- Balanced  
- Overperforming  

### Logistic Regression
- Baseline model  
- High interpretability  

### Random Forest
- Captures complex patterns  
- More robust predictions  

### Majority Voting
Final prediction based on:
- K-Means  
- Logistic Regression  
- Random Forest  

---

## 📈 Model Performance

| Model | Accuracy |
|------|---------|
| Logistic Regression | 98.3% |
| Random Forest | 96.2% |

---

## 🎮 Difficulty Recommendation

| Player Group | Recommendation |
|------------|---------------|
| Struggling | Decrease Difficulty |
| Balanced | Keep Same |
| Overperforming | Increase Difficulty |

---

## 🌐 Streamlit Application

### Features
- Single player prediction  
- Manual input  
- CSV batch upload  
- Download results  
- Charts and insights



## 📌 Key Findings

- Most players are Struggling or Balanced  
- K/D ratio and accuracy are key indicators  
- SQL helps identify drop-off risk and engagement patterns  
- ML models classify players effectively  
- Recommendations improve gameplay balance  
  

---

## 🔮 Future Improvements

- Real-time difficulty adjustment  
- Reinforcement learning  
- Player feedback integration  
- Behavioral analytics  

---

## 👨‍💻 Author

Shreyas Keragodu Vishwanath  
MSc Data Analytics  

---

## ⭐ If you like this project
Give it a ⭐ on GitHub!
