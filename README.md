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

Dataset
   ↓
SQL Analysis Layer
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
K-Means Clustering
   ↓
Logistic Regression + Random Forest
   ↓
Majority Voting
   ↓
Difficulty Recommendation
   ↓
Streamlit Application
