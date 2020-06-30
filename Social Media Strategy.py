# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:25:38 2020

@author: er1cc
"""

import nltk
from nltk import word_tokenize
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
from scipy import stats
import statsmodels.formula.api as sm
import statsmodels.stats.outliers_influence as sm_influence
pip install vaderSentiment
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_columns',500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 500)

# PART 1, CLEANING AND PREPARING FACEBOOK EXCEL SHEET FOR ANALYSIS
# Importing Facebook excel file

facebook = pd.read_excel(r'C:/Users/_____/Facebook Data.xlsx', usecols=9)

# For the facebook DataFrame, fill nulls for columns that involved engagement
null_columns = {"Other Clicks": 0, "Link Clicks": 0, "Likes": 0, "Comments": 0, "Shares": 0}
facebook.fillna(null_columns, axis=0, inplace=True)
facebook.isnull().sum(axis=0)

# Extract only the hour (military time)
facebook['Time Posted'] = facebook['Date'].dt.round('60min').dt.hour

# Create engagement and engagement rate
facebook['Engagement'] = facebook['Other Clicks'] + facebook['Link Clicks'] + facebook['Likes'] + facebook['Comments'] + facebook['Shares']
facebook['Engagement Rate'] = (facebook['Engagement'] / facebook['Impressions']).round(2)

# PART 2: Create engagement rates per topic of post column      
a = facebook.groupby(facebook['Topic of Post'], as_index=False)['Engagement Rate'].mean().round(2)
a = facebook.groupby(facebook['Topic of Post'], as_index=False)['Engagement Rate'].mean().round(2)
a.rename(columns = {"Engagement Rate": 'Average Engagement Rate per Post Topic'}, inplace=True)

facebook = pd.merge(facebook, a, how='inner', on=['Topic of Post'])

# Analysis for PART 2 (Topic 3 has the most engagement)
plt.scatter(facebook['Topic of Post'], facebook['Average Engagement Rate per Post Topic'])
plt.title('Average Engagement Rate per Post Topic')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Specific to form of engagement
facebook['Rate Post Clicks'] = ((facebook['Link Clicks'] + facebook['Other Clicks']) / facebook['Impressions']).round(2)
facebook['Rate Likes'] = (facebook['Likes'] / facebook['Impressions']).round(2)
facebook['Rate Comments'] = (facebook['Comments'] / facebook['Impressions']).round(2)
facebook['Rate Shares'] = (facebook['Shares'] / facebook['Impressions']).round(2)
        
b = facebook.groupby(facebook['Topic of Post'], as_index=False).agg({'Rate Post Clicks' : 'mean', 'Rate Likes' : 'mean', 'Rate Comments' : 'mean', 'Rate Shares' : 'mean'}).round(2)
b = facebook.groupby(facebook['Topic of Post'], as_index=False).agg({'Rate Post Clicks' : 'mean', 'Rate Likes' : 'mean', 'Rate Comments' : 'mean', 'Rate Shares' : 'mean'}).round(2)
b.rename(columns = {'Rate Post Clicks' : 'Average Rate Post Clicks', 'Rate Likes' : 'Average Rate Likes', 'Rate Comments' : 'Average Rate Comments', 'Rate Shares' : 'Average Rate Shares'}, inplace=True)        

facebook = pd.merge(facebook, b, how='inner', on=['Topic of Post'])
facebook.dtypes

# Analysis for Clicks (Topic 3 has the most engagement for clicks)
plt.scatter(facebook['Topic of Post'], facebook['Average Rate Post Clicks'])
plt.title('Average Engagement Rate based on Post Clicks')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Likes (Topic 3 have the most engagement for likes)
plt.scatter(facebook['Topic of Post'], facebook['Average Rate Likes'])
plt.title('Average Engagement Rate based on Post Likes')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Comments (Topic 1, 3, and 4 have the most engagement for likes)
plt.scatter(facebook['Topic of Post'], facebook['Average Rate Comments'])
plt.title('Average Engagement Rate based on Post Comments')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Shares (Topic 3 has the most engagement for likes)
plt.scatter(facebook['Topic of Post'], facebook['Average Rate Shares'])
plt.title('Average Engagement Rate based on Post Shares')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 3: Create engagement rates per type of post column      
c = facebook.groupby(facebook['Type'], as_index=False)['Engagement Rate'].mean().round(2)
c = facebook.groupby(facebook['Type'], as_index=False)['Engagement Rate'].mean().round(2)
c.rename(columns = {"Engagement Rate": 'Average Engagement Rate per Post Type'}, inplace=True)

facebook = pd.merge(facebook, c, how='inner', on=['Type'])

# Analysis for PART 3 (Links have the most engagement)
plt.scatter(facebook['Type'], facebook['Average Engagement Rate per Post Type'])
plt.title('Average Engagement Rate per Post Type')
plt.xlabel('Type of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 4: Best Date and Time to post on facebook
facebook['Day of the Week'] = facebook['Date'].dt.weekday_name

# Create engagement rate per day of week columns
d = facebook.groupby(facebook['Day of the Week'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
d = facebook.groupby(facebook['Day of the Week'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
d.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Day of Week'}, inplace=True) 

facebook = pd.merge(facebook, d, how='inner', on=['Day of the Week'])

# Analysis visual for day of the week (Thursdays have the most engagement)
plt.scatter(facebook['Day of the Week'], facebook['Average Engagement Rate by Day of Week'])
plt.title('Average Engagement Rate by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Create engagement rate per time of day
e = facebook.groupby(facebook['Time Posted'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
e = facebook.groupby(facebook['Time Posted'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
e.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Time'}, inplace=True) 

facebook = pd.merge(facebook, e, how='inner', on=['Time Posted'])
facebook.dtypes

# Analysis Visual (Best time is midnight)
plt.scatter(facebook['Time Posted'], facebook['Average Engagement Rate by Time'])
plt.title('Average Engagement Rate based on Time of Post')
plt.xlabel('Time of Day')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 5: How many times a day result in the most engagement for facebook

facebook['Date-No Time'] = facebook['Date'].dt.date
facebook['Number of Posts Per Day'] = facebook.groupby(['Date-No Time'])['Date-No Time'].transform('count')

f = facebook.groupby(facebook['Number of Posts Per Day'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
f = facebook.groupby(facebook['Number of Posts Per Day'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
f.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Number of Posts Per Day'}, inplace=True)

facebook = pd.merge(facebook, f, how='inner', on=['Number of Posts Per Day'])

# Analysis Visual (1 and 4 posts result in the most engagement)
plt.scatter(facebook['Number of Posts Per Day'], facebook['Average Engagement Rate by Number of Posts Per Day'])
plt.title('Average Engagement Rate based on Number of Posts Per Day')
plt.xlabel('Number of Posts Per Day')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 6: Regression for Facebook
facebook_copy = facebook.copy()

print((facebook_copy.isnull()).sum())

# Test for Normality
sns.boxplot(x=facebook_copy['Engagement Rate'])
plt.figure()
sns.distplot(facebook_copy['Engagement Rate'], kde=False, fit=stats.norm)
plt.figure()

# Modifications for normality
facebook_copy['Modified Engagement Rate'] = np.sqrt(facebook_copy['Engagement Rate'])
sns.distplot(facebook_copy['Modified Engagement Rate'], kde=False, fit=stats.norm)
plt.figure()

# Test for Normality with Modified
sns.boxplot(x=facebook_copy['Modified Engagement Rate'])
plt.figure()

# Winsorize Data
print((facebook_copy['Modified Engagement Rate'] < facebook_copy['Modified Engagement Rate'].quantile(q=0.02)).sum())
print((facebook_copy['Modified Engagement Rate'] > facebook_copy['Modified Engagement Rate'].quantile(q=0.98)).sum())

facebook_copy['Modified Engagement Rate'] = np.where(facebook_copy['Modified Engagement Rate'] < facebook_copy['Modified Engagement Rate'].quantile(q=0.02), facebook_copy['Modified Engagement Rate'].quantile(q=0.02), np.where(facebook_copy['Modified Engagement Rate'] > facebook_copy['Modified Engagement Rate'].quantile(q=0.98), facebook_copy['Modified Engagement Rate'].quantile(q=0.98), facebook_copy['Modified Engagement Rate']))

# Test for Normality with Modified after winsorizing
sns.boxplot(x=facebook_copy['Modified Engagement Rate'])
plt.figure()

# Correlation (dependent: Type(create dummy), Topic, Time Posted, Day of the Week(create dummy), Number of Posts per Day)
facebook_copy.dtypes
facebook_copy.corr()['Modified Engagement Rate']

# Rename all columns to be regression friendly
facebook_copy.columns = [c.replace(' ', '_') for c in facebook_copy.columns]

# Regression
model_results = sm.ols(formula = 'Modified_Engagement_Rate ~ Type + Day_of_the_Week + Number_of_Posts_Per_Day + C(Time_Posted) + C(Topic_of_Post)', data=facebook_copy).fit()                          
print(model_results.summary())

# Convert data types for noncategorical variables
facebook_copy = facebook_copy.astype({'Time_Posted': object, 'Topic_of_Post': object})

# Dummies for categorical variables
facebook_copy = pd.get_dummies(facebook_copy, columns=['Day_of_the_Week'], drop_first=True)
facebook_copy = pd.get_dummies(facebook_copy, columns=['Time_Posted'], drop_first=True)
facebook_copy = pd.get_dummies(facebook_copy, columns=['Topic_of_Post'], drop_first=True)

# Regression Part 2
model_results = sm.ols(formula = 'Modified_Engagement_Rate ~ Day_of_the_Week_Saturday + Time_Posted_12 + Topic_of_Post_7 + Topic_of_Post_9', data=facebook_copy).fit()                          
print(model_results.summary())

#Test Assumption of Multicollinearity
#Check VIF Factor for variables
myX = facebook_copy[['Day_of_the_Week_Saturday', 'Time_Posted_12', 'Topic_of_Post_7', 'Topic_of_Post_9']]
myX = myX.dropna()
vif = pd.DataFrame()
vif["VIF Factor"] = [sm_influence.variance_inflation_factor(myX.values, i) for i in range(myX.shape[1])]
vif["Variable"]=myX.columns
print(vif.round(2)) 

#Check if the errors are normally distributed
facebook_copy['residuals'] = model_results.resid
facebook_copy['predicted'] = model_results.fittedvalues
sns.distplot(facebook_copy.residuals, kde=False, fit=stats.norm)
plt.figure()

#Test Assumption of Heteroskedasticity
# Plot residuals by predicted
facebook_copy['residuals'] = model_results.resid
facebook_copy['predicted'] = model_results.fittedvalues
plt.scatter(facebook_copy.predicted, facebook_copy.residuals)
plt.title('Residuals by Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.figure()

# Plot predicted by actual
plt.scatter(facebook_copy.predicted, facebook_copy.Modified_Engagement_Rate)
plt.title("Actual by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actuals")

# PART 7: NLP for Facebook

nlp_df = facebook.groupby(facebook['Topic of Post'], as_index=True)['Post Message'].apply(' '.join).reset_index()

stopwords = open(r"C:\Users\__________\StopWords.txt", encoding='utf-8').readlines()
stopwords = [s.rstrip() for s in stopwords]

Text_List = []
Text_List_Strings = []

for lst in nlp_df['Post Message']:
    lst = re.sub(r"http.*? ", " ", lst)
    lst = re.sub(r"#\w{1,}", " ", lst)
    lst = re.sub(r"dhw", " ", lst)
    lst = re.sub(r"[^A-Za-z ]", "", lst)
    lst = re.sub(r" \w{1} ", " ", lst)
    lst = lst.upper()
    lst = re.sub(r"DANIEL", " ", lst)
    lst = re.sub(r"WOLF", " ", lst)
    lst = re.sub(r"\s{2,}", " ", lst)
    lst = word_tokenize(lst)
    lst = [word for word in lst if word not in stopwords]
    Text_List.append(lst)
    Text_List_Strings.append(' '.join(lst))

nlp_df['Clean Post Message String'] = Text_List_Strings
nlp_df['Clean Post Message'] = Text_List

# Part 8: Sentiment Analysis for Facebook

analyzer = SentimentIntensityAnalyzer()

for string in nlp_df['Clean Post Message String']:
    vs = analyzer.polarity_scores(string)
    print(vs) 

# Wordcloud
cloud_stopwords = set(STOPWORDS) 

for string in nlp_df['Clean Post Message String']:
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = cloud_stopwords, 
                min_font_size = 10).generate(string) 
  
# plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show() 

# PART 9: CLEANING AND PREPARING TWITTER EXCEL SHEET FOR ANALYSIS
# Importing Twitter excel file
twitter_draft = pd.read_excel(r'C:/Users/_____________/Twitter Data.xlsx', usecols=16)

# Converging posts that were meant to be continuous 

twitter_draft.drop(columns=['Tweet id', 'Tweet permalink'], inplace=True)
twitter_draft.dtypes

twitter = twitter_draft.groupby(twitter_draft['Continuous Post'], as_index=False).agg({'Tweet text' : 'sum', 'Date' : 'min', 'Type' : 'mean', 'Topic of Post' : lambda x: x.mode(), 'Impressions' : 'sum', 'Engagements' : 'sum', 'Engagement Rate' : 'mean', 'Retweets' : 'sum', 'Replies' : 'sum', 'Likes' : 'sum', 'user profile clicks' : 'sum', 'url clicks' : 'sum', 'hashtag clicks' : 'sum', 'detail expands' : 'sum'})
twitter.drop(['Continuous Post'], axis=1, inplace=True)

# Round engagement rate to two decimal places
twitter['Engagement Rate'] = twitter['Engagement Rate'].round(2)
print(twitter.dtypes)

# Creating columns for time and date (changing time to pacific time)
twitter['Date']
twitter['Date'] = pd.to_datetime(twitter['Date'], format="%Y-%m-%d %H:%M +0000")
twitter['Date'] = twitter['Date'] - timedelta(hours=8)
twitter['Date'] = twitter['Date'].dt.round('60min')

twitter['Date-No Time'] = twitter['Date'].dt.date
twitter['Date-No Time'] = pd.to_datetime(twitter['Date-No Time'], format="%Y-%m-%d")
twitter['Time'] = twitter['Date'].dt.time
twitter['Time'] = pd.to_datetime(twitter['Time'], format="%H:%M:%S").dt.hour

# PART 10: Topics that result in the most downloads

a = twitter.groupby(twitter['Topic of Post'], as_index=False)['Engagement Rate'].mean().round(2)
a = twitter.groupby(twitter['Topic of Post'], as_index=False)['Engagement Rate'].mean().round(2)
a.rename(columns = {"Engagement Rate": 'Average Engagement Rate per Post Topic'}, inplace=True)

twitter = pd.merge(twitter, a, how='inner', on=['Topic of Post'])

# Analysis for PART 7 (Topic 8, 4 has the most engagement)
plt.scatter(twitter['Topic of Post'], twitter['Average Engagement Rate per Post Topic'])
plt.title('Average Engagement Rate per Post Topic')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Specific to form of engagements
twitter['Rate Retweets'] = (twitter['Retweets'] / twitter['Impressions']).round(2)
twitter['Rate Replies'] = (twitter['Replies'] / twitter['Impressions']).round(2)
twitter['Rate Likes'] = (twitter['Likes'] / twitter['Impressions']).round(2)
twitter['Rate user profile clicks'] = (twitter['user profile clicks'] / twitter['Impressions']).round(2)
twitter['Rate url clicks'] = (twitter['url clicks'] / twitter['Impressions']).round(2)
twitter['Rate hashtag clicks'] = (twitter['hashtag clicks'] / twitter['Impressions']).round(2)
twitter['Rate detail expands'] = (twitter['detail expands'] / twitter['Impressions']).round(2)     

b = twitter.groupby(twitter['Topic of Post'], as_index=False).agg({'Rate Retweets' : 'mean', 'Rate Replies' : 'mean', 'Rate Likes' : 'mean', 'Rate user profile clicks' : 'mean', 'Rate url clicks' : 'mean', 'Rate hashtag clicks' : 'mean', 'Rate detail expands' : 'mean'}).round(2)
b = twitter.groupby(twitter['Topic of Post'], as_index=False).agg({'Rate Retweets' : 'mean', 'Rate Replies' : 'mean', 'Rate Likes' : 'mean', 'Rate user profile clicks' : 'mean', 'Rate url clicks' : 'mean', 'Rate hashtag clicks' : 'mean', 'Rate detail expands' : 'mean'}).round(2)
b.rename(columns = {'Rate Retweets' : 'Average Rate Retweets', 'Rate Replies' : 'Average Rate Replies', 'Rate Likes' : 'Average Rate Likes', 'Rate user profile clicks' : 'Average Rate User Profile Clicks', 'Rate url clicks' : 'Average Rate Url Clicks', 'Rate hashtag clicks' : 'Average Rate Hashtag Clicks', 'Rate detail expands' : 'Average Rate Detail Expands'}, inplace=True)        

twitter = pd.merge(twitter, b, how='inner', on=['Topic of Post'])

# Analysis for Retweets (No topic has the most engagement for retweets)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Retweets'])
plt.title('Average Engagement Rate based on Post Retweets')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Replies (Topic 3, 10 have the most engagement for replies)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Replies'])
plt.title('Average Engagement Rate based on Post Replies')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Likes (Topic 1, 2, 4, 11 have the most engagement for likes)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Likes'])
plt.title('Average Engagement Rate based on Post Likes')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for User Profile CLicks (Topic 3, 4, 10 has the most engagement for User Profile Clicks)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate User Profile Clicks'])
plt.title('Average Engagement Rate based on Post User Profile Clicks')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Url Clicks (Topic 8, 9 has the most engagement for Url Clicks)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Url Clicks'])
plt.title('Average Engagement Rate based on Post Url Clicks')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Hashtag Clicks (No topic has the most engagement for Hashtag Clicks)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Hashtag Clicks'])
plt.title('Average Engagement Rate based on Post Hashtag Clicks')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Analysis for Detail Expands (Topic 4 has the most engagement for Detail Expands)
plt.scatter(twitter['Topic of Post'], twitter['Average Rate Detail Expands'])
plt.title('Average Engagement Rate based on Post Detail Expands')
plt.xlabel('Topic of Post')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 11: Create engagement rates per type of post column
c = twitter.groupby(twitter['Type'], as_index=False)['Engagement Rate'].mean().round(2)
c = twitter.groupby(twitter['Type'], as_index=False)['Engagement Rate'].mean().round(2)
c.rename(columns = {"Engagement Rate": 'Average Engagement Rate per Post Type'}, inplace=True)

twitter = pd.merge(twitter, c, how='inner', on=['Type'])

# Analysis for PART 3 (Type 0 and 1 have the most engagement)
plt.scatter(twitter['Type'], twitter['Average Engagement Rate per Post Type'])
plt.title('Average Engagement Rate per Post Type')
plt.xlabel('Type of Post')
plt.xticks(np.arange(0, 3, step=1))
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 12: Times and days in which twitter has the most engagement
twitter['Day of the Week'] = twitter['Date'].dt.weekday_name

# Create engagement rate per day of week columns
d = twitter.groupby(twitter['Day of the Week'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
d = twitter.groupby(twitter['Day of the Week'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
d.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Day of Week'}, inplace=True) 

twitter = pd.merge(twitter, d, how='inner', on=['Day of the Week'])

# Analysis visual for day of the week (Saturdays have the most engagement)
plt.scatter(twitter['Day of the Week'], twitter['Average Engagement Rate by Day of Week'])
plt.title('Average Engagement Rate by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Engagement Rate')
plt.figure()

# Create engagement rate per time of day
e = twitter.groupby(twitter['Time'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
e = twitter.groupby(twitter['Time'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
e.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Time'}, inplace=True) 

twitter = pd.merge(twitter, e, how='inner', on=['Time'])

# Analysis Visual (Best time is 19 then 21)
plt.scatter(twitter['Time'], twitter['Average Engagement Rate by Time'])
plt.title('Average Engagement Rate based on Time of Post')
plt.xlabel('Time of Day')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 13: How many times per day to post on twitter

twitter['Number of Posts Per Day'] = twitter.groupby(['Date-No Time'])['Date-No Time'].transform('count')

f = twitter.groupby(twitter['Number of Posts Per Day'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
f = twitter.groupby(twitter['Number of Posts Per Day'], as_index=False).agg({'Engagement Rate' : 'mean'}).round(2)
f.rename(columns = {'Engagement Rate' : 'Average Engagement Rate by Number of Posts Per Day'}, inplace=True)

twitter = pd.merge(twitter, f, how='inner', on=['Number of Posts Per Day'])

# Analysis Visual (2 and 19 posts result in the most engagement)
plt.scatter(twitter['Number of Posts Per Day'], twitter['Average Engagement Rate by Number of Posts Per Day'])
plt.title('Average Engagement Rate based on Number of Posts Per Day')
plt.xlabel('Number of Posts Per Day')
plt.ylabel('Average Engagement Rate')
plt.figure()

# PART 14: Regression for Twitter
twitter_copy = twitter.copy()

print((twitter_copy.isnull()).sum())

# Test for Normality
sns.boxplot(x=twitter_copy['Engagement Rate'])
plt.figure()
sns.distplot(twitter_copy['Engagement Rate'], kde=False, fit=stats.norm)
plt.figure()

# Modifications for normality
twitter_copy['Modified Engagement Rate'] = np.sqrt(twitter_copy['Engagement Rate'])
sns.distplot(twitter_copy['Modified Engagement Rate'], kde=False, fit=stats.norm)
plt.figure()

# Test for Normality with Modified
sns.boxplot(x=twitter_copy['Modified Engagement Rate'])
plt.figure()

# Winsorize Data
print((twitter_copy['Modified Engagement Rate'] < twitter_copy['Modified Engagement Rate'].quantile(q=0.02)).sum())
print((twitter_copy['Modified Engagement Rate'] > twitter_copy['Modified Engagement Rate'].quantile(q=0.98)).sum())

twitter_copy['Modified Engagement Rate'] = np.where(twitter_copy['Modified Engagement Rate'] < twitter_copy['Modified Engagement Rate'].quantile(q=0.02), twitter_copy['Modified Engagement Rate'].quantile(q=0.02), np.where(twitter_copy['Modified Engagement Rate'] > twitter_copy['Modified Engagement Rate'].quantile(q=0.98), twitter_copy['Modified Engagement Rate'].quantile(q=0.98), twitter_copy['Modified Engagement Rate']))

# Test for Normality with Modified after winsorizing
sns.boxplot(x=twitter_copy['Modified Engagement Rate'])
plt.figure()

# Correlation (dependent: Type(create dummy), Topic, Time Posted, Day of the Week(create dummy), Number of Posts per Day)
twitter_copy.dtypes
twitter_copy.corr()['Modified Engagement Rate']

# Rename all columns to be regression friendly
twitter_copy.columns = [c.replace(' ', '_') for c in twitter_copy.columns]

# Regression
model_results = sm.ols(formula = 'Modified_Engagement_Rate ~ Type + Day_of_the_Week + Number_of_Posts_Per_Day + C(Time) + C(Topic_of_Post)', data=twitter_copy).fit()                          
print(model_results.summary())

# Convert data types for noncategorical variables
twitter_copy = twitter_copy.astype({'Time': object, 'Day_of_the_Week': object})

# Dummies for categorical variables
twitter_copy = pd.get_dummies(twitter_copy, columns=['Day_of_the_Week'], drop_first=True)
twitter_copy = pd.get_dummies(twitter_copy, columns=['Time'], drop_first=True)

# Regression Part 2
model_results = sm.ols(formula = 'Modified_Engagement_Rate ~ Day_of_the_Week_Sunday + Time_19 + Time_21 + Time_22', data=twitter_copy).fit()                          
print(model_results.summary())

#Test Assumption of Multicollinearity
#Check VIF Factor for variables
myX = twitter_copy[['Number_of_Posts_Per_Day', 'Topic_of_Post']]
myX = myX.dropna()
vif = pd.DataFrame()
vif["VIF Factor"] = [sm_influence.variance_inflation_factor(myX.values, i) for i in range(myX.shape[1])]
vif["Variable"]=myX.columns
print(vif.round(2)) 

#Check if the errors are normally distributed
twitter_copy['residuals'] = model_results.resid
twitter_copy['predicted'] = model_results.fittedvalues
sns.distplot(twitter_copy.residuals, kde=False, fit=stats.norm)
plt.figure()

#Test Assumption of Heteroskedasticity
# Plot residuals by predicted
twitter_copy['residuals'] = model_results.resid
twitter_copy['predicted'] = model_results.fittedvalues
plt.scatter(twitter_copy.predicted, twitter_copy.residuals)
plt.title('Residuals by Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.figure()

# Plot predicted by actual
plt.scatter(twitter_copy.predicted, twitter_copy.Modified_Engagement_Rate)
plt.title("Actual by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actuals")

# PART 15: NLP for Twitter

nlp_df = twitter.groupby(twitter['Topic of Post'], as_index=True)['Tweet text'].apply(' '.join).reset_index()

Text_List = []
Text_List_Strings = []

for lst in nlp_df['Tweet text']:
    lst = re.sub(r"@\w{1,}", " ", lst)
    lst = re.sub(r"http.*? ", " ", lst)
    lst = re.sub(r"#\w{1,}", " ", lst)
    lst = re.sub(r"dhw", " ", lst)
    lst = re.sub(r"[^A-Za-z ]", "", lst)
    lst = re.sub(r" \w{1} ", " ", lst)
    lst = lst.upper()
    lst = re.sub(r"DANIEL", " ", lst)
    lst = re.sub(r"DAN", " ", lst)
    lst = re.sub(r"WOLF", " ", lst)
    lst = re.sub(r"\s{2,}", " ", lst)
    lst = word_tokenize(lst)
    lst = [word for word in lst if word not in stopwords]
    Text_List.append(lst)
    Text_List_Strings.append(' '.join(lst))

nlp_df['Clean Tweet Text String'] = Text_List_Strings
nlp_df['Clean Tweet Text'] = Text_List

analyzer = SentimentIntensityAnalyzer()

for string in nlp_df['Clean Tweet Text String']:
    vs = analyzer.polarity_scores(string)
    print(vs) 
    
# Wordcloud
cloud_stopwords = set(STOPWORDS) 

for string in nlp_df['Clean Tweet Text String']:
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = cloud_stopwords, 
                min_font_size = 10).generate(string) 
  
# plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show() 