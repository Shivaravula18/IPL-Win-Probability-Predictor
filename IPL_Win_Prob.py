# IPL Probability Predictor

#This is a kaggle dataset where we have 5 Files but we are focused only on 2 Files :
#1. matches.csv $\rightarrow$ Has each match analysis
#2. deliveries.csv $\rightarrow$ Has each ball analysis.

##### Data Set Link : https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

# Import the libraries for data cleaning
import numpy as np
import pandas as pd

# import the data
match_df = pd.read_csv('Data/matches.csv')
delivery_df = pd.read_csv('Data/deliveries.csv')

match_df.head()

match_df.shape

match_df.columns

match_df.drop(['toss_winner','toss_decision','player_of_match','umpire1', 'umpire2','umpire3'],axis=1,inplace=True)

match_df.head()

match_df.shape

delivery_df.head()

delivery_df.columns

delivery_df.drop(['bowler','is_super_over','dismissal_kind','fielder','wide_runs','bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',],axis=1,inplace=True)

delivery_df.head()

delivery_df.shape

"""- batsman_runs --> Runs made by bats man
- total_runs --> Runs scored
"""

total_score_df = (delivery_df.groupby(['match_id','inning']).sum()['total_runs'].reset_index())

total_score_df

total_score_df = total_score_df[total_score_df['inning']==1]

total_score_df

match_df

match_df = (match_df.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id'))

match_df

teams = list(match_df['team1'].unique())

teams

"""-  'Sunrisers Hyderabad'Mu- mbai Indians',
  --> Gujarat Titans '- Gujarat Lions',
 'Rising  --> No MoreP- une Supergiant',
 'Royal Challe- ngers Bangalore',
 'Kolka- ta Knight Riders',
 --> 'Delhi Capitals' -' Delhi Daredevils',
-
  'Kings XI Punjab',
 '-C hennai Super Kings',-

 'Rajasthan Royals',
'Sunrisers Hyderabad' -' Deccan  --Chargers',
 --> No More
-  'Kochi Tuskers K --> No Moree-r ala',
 'Pune Warriors',
 -' Rising Pune Supergiants',
 'Delhi Capitals'
"""

teams = ['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Gujarat Titans',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals']

# Replace Delhi Daredevils with Delhi capitals
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

match_df['team1'] = match_df['team1'].str.replace('Gujarat Lions', 'Gujarat Titans')
match_df['team2'] = match_df['team2'].str.replace('Gujarat Lions', 'Gujarat Titans')

match_df['team1'].unique()

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df['team1'].unique()

match_df.shape

match_df.head()

# dl_applied is 0 means normal match and 1 means duckworth lewis rules are applied
match_df['dl_applied'].value_counts()

# Removing dl_applied = 1 records or rows
match_df = match_df[match_df['dl_applied'] == 0]

match_df.shape

match_df.columns

match_df

match_df = match_df[['match_id','city','winner','total_runs']]

delivery_df.head()

delivery_df = delivery_df[delivery_df['inning'] == 2 ]

delivery_df.info()

delivery_df = match_df.merge(delivery_df, on='match_id')

delivery_df

delivery_df.columns

delivery_df.rename(columns={'total_runs_x':'total_runs','total_runs_y':'Ball_score'},inplace=True)

delivery_df

delivery_df['Score'] = delivery_df[['match_id','Ball_score']].groupby('match_id').cumsum()['Ball_score']

delivery_df[['Ball_score','Score']]

# New columns which has target
delivery_df['target_left'] = (delivery_df['total_runs'] + 1) - delivery_df['Score']

delivery_df

# 1 over 6 balls
# ipl has 20 over match 20 * 6 = 120
delivery_df['Remaining Balls'] = (120 - ( (delivery_df['over'] - 1)*6 + delivery_df['ball']))

delivery_df

# How many  wickets are left out

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna('0')

delivery_df

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == '0' else '1').astype('int64')

delivery_df

delivery_df[['match_id','player_dismissed']]

delivery_df['Wickets'] = delivery_df[['match_id','player_dismissed']].groupby('match_id').cumsum()['player_dismissed'].values

delivery_df

delivery_df['Wickets'] = 10 - delivery_df['Wickets']

delivery_df

"""Current Run Rate

$$Current \ run \ rate = \frac{Runs \ scored}{Number \ of \ Overs \ completed}$$
"""

delivery_df['crr'] = (delivery_df['Score'])*6/(120 - delivery_df['Remaining Balls'])

delivery_df

"""Required Run Rate

$$Required \ run \ rate = \frac{Remaining \ score \ or \ Target}{Number \ of \ Overs \ left}$$
"""

delivery_df['rrr'] = (delivery_df['target_left'])*6/(delivery_df['Remaining Balls'])

delivery_df

# Creating a column result where 1 means won 0 means lost
def result(row):
    if row['batting_team'] == row['winner']:
        return 1
    else:
        return 0

delivery_df['result'] = delivery_df.apply(result,axis=1)

delivery_df

delivery_df.columns

Model_df = delivery_df[['batting_team','bowling_team','city','Score','Wickets','Remaining Balls','target_left','crr','rrr','result']]

Model_df

Model_df.isnull().sum()

# Remove the records where city is missing
Model_df = Model_df.replace([np.inf, -np.inf], np.nan)
Model_df[Model_df['city'].isna()]

Model_df = Model_df.dropna()

Model_df.isnull().sum()

Model_df.describe()

Model_df = Model_df[Model_df['Remaining Balls'] != 0]

Model_df.describe()

Model_df = Model_df.sample(Model_df.shape[0])

Model_df.sample()

# Splitting the data into features and target

X = Model_df.iloc[:,:-1]

Y = Model_df.iloc[:,-1]

X

Y

# Splitting the data to train and test :
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

X_train

X_test

Y_train

Y_test

# Label encoding
# One hot encoder
# tranform non numberical data to numerical

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])], remainder = 'passthrough')

trf

# Building the model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train,Y_train)

Y_prediction = pipe.predict(X_test)

Y_prediction

# Accuracy score
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,Y_prediction)

# Exporting the model
import pickle as pkl

pkl.dump(pipe,open('model.pkl','wb'))

teams

pkl.dump(teams,open('team.pkl','wb'))

cities = list(Model_df['city'].unique())

cities

pkl.dump(cities,open('city.pkl','wb'))

Model_df.columns

