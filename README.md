# IPL Win Probability Predictor

A machine learning-based web app that predicts the winning probability of an IPL cricket team during a match using live match data.

---

## Project Overview

This project uses historical IPL match and ball-by-ball data to train a logistic regression model to estimate the chances of a batting team winning at any point in the second innings of a match.

The app allows users to input match details like batting team, bowling team, city, current score, wickets fallen, overs completed, and target score. It then predicts and displays the probability of each team winning.

---

## Dataset

- Source: [Kaggle IPL Dataset](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
- Main files used:
  - `matches.csv` — match-level details
  - `deliveries.csv` — ball-by-ball details

---

## Features

- Batting team
- Bowling team
- Host city
- Current score
- Wickets fallen
- Overs completed
- Target score

Derived features like current run rate, required run rate, remaining balls, and wickets left are also used.

---

## Tech Stack

- Python 3.10+
- Pandas, NumPy (data processing)
- Scikit-learn (model training)
- Streamlit (web app)
- Pickle (model serialization)

---

## How to Run

1. Clone this repository or download the files.

2. Ensure you have Python 3.10+ installed.

3. Install dependencies:

```bash
pip install -r requirements.txt
