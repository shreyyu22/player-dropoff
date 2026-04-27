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

Derived Feature:
-```text
--accuracy = hits / shots

## Project Architecture

-Dataset
   ↓
-SQL Analysis Layer
   ↓
-Exploratory Data Analysis (EDA)
   ↓
-Feature Engineering
   ↓
-K-Means Clustering
   ↓
-Logistic Regression + Random Forest
   ↓
-Majority Voting
   ↓
-Difficulty Recommendation
   ↓
-Streamlit Application

## SQL Analysis

-SQL is used as a core analytical layer to extract insights from player data.

Key SQL Insights

1. Performance Ranking
Top players have significantly higher performance scores
Quartile segmentation separates elite players from average players
High-ranking players require increased difficulty

2. Engagement vs Performance
Low engagement players are mostly struggling
High engagement players perform more consistently
Low engagement players are at higher drop-off risk

3. Win Rate Analysis
Win rate shows player consistency
Combining win rate and score per minute improves performance evaluation

4. Accuracy Analysis
Accuracy strongly correlates with player skill
High accuracy players perform better overall
Low accuracy players may need easier difficulty

5. Drop-Off Risk Detection
Low playtime + low K/D ratio → high drop-off risk
Early intervention can improve retention

6. Recommendation Validation
Struggling → Decrease Difficulty
Balanced → Keep Same
Overperforming → Increase Difficulty
Recommendations align with player behavior

7. Outlier Detection
High outliers = expert players
Low outliers = inactive/weak players

8. Player Scoring Index
Multi-feature scoring gives better ranking
Players classified into Beginner, Casual, Competitive, Elite
