import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('popular_10000_movies_tmdb.csv')

##################################################################
# Data Preprocessing

# Remove unnecessary columns
cols_to_drop = ['overview', 'tagline']
data = data.drop(cols_to_drop, axis=1)


# #Convert the release_date column to datetime format:
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year
data['release_month'] = data['release_date'].dt.month
data['release_day'] = data['release_date'].dt.day
data = data[data['release_year'] <= 2023]


# Check for missing values
missing_values = data.isnull().sum()
data = data.dropna()

budget_col = data['budget']
budget_col.replace(to_replace=0, value=budget_col.mean(), inplace=True)
column_name = 'budget'


revenue_col = data['revenue']
revenue_col.replace(to_replace=0, value=revenue_col.mean(), inplace=True)
column_name = 'revenue'


runtime_col = data['runtime']
runtime_col.replace(to_replace=0, value=runtime_col.mean(), inplace=True)
column_name = 'runtime'


popularity_col = data['popularity']
popularity_col.replace(to_replace=0, value=popularity_col.mean(), inplace=True)
column_name = 'popularity'


vote_count_col = data['vote_count']
vote_count_col.replace(to_replace=0, value=vote_count_col.mean(), inplace=True)
column_name = 'vote_count'

vote_average_col = data['vote_average']
vote_average_col.replace(
    to_replace=0, value=vote_average_col.mean(), inplace=True)
column_name = 'vote_average'


# Replace '[]' values with the second most frequent value
column_name = 'production_companies'
frequencies = data[column_name].value_counts()
second_most_frequent_value = frequencies.index[1]
data[column_name] = data[column_name].replace('[]', second_most_frequent_value)


# Replace '[]' values with the second most frequent value
column_name = 'genres'
frequencies = data[column_name].value_counts()
second_most_frequent_value = frequencies.index[1]
data[column_name] = data[column_name].replace('[]', second_most_frequent_value)



# Calculate the median of the target variable
# Define the columns to consider for outlier replacement
columns_to_check = ['vote_average', 'vote_count', 'popularity', 'runtime', 'budget']
# Loop through each column
for column in columns_to_check:
    # Calculate the median of the column
    median = np.median(data[column])

    # Define the threshold value as a multiple of the median (e.g., 3 times the median)
    threshold = 3 * median

    # Identify the outliers
    outliers = data[data[column] > threshold]

    # # Replace the outliers with the median value as an integer
    data.loc[data[column] > threshold, column] = int(median)

data['genres'] = data['genres'].str.replace('[', '', regex=False).str.replace(
    ']', '', regex=False).str.replace("'\s*|\s*'", '', regex=True)
data['production_companies'] = data['production_companies'].str.replace(
    '[', '', regex=False).str.replace(']', '', regex=False).str.replace("'\s*|\s*'", '', regex=True)

#####################################################################################
# Analysis and Visualization by Python

# max popularity and title
most_popular_film = data.loc[data['popularity'].idxmax(), 'title']
# min popularity and title
min_popularity_film = data.loc[data['popularity'].idxmin(), 'title']
# average_popularity
max_popularity = data['popularity'].max()
# min popularity
min_popularity = data['popularity'].min()

# max runtime and title
most_runtime_film = data.loc[data['runtime'].idxmax(), 'title']
# min runtime and title
most_runtime_film = data.loc[data['runtime'].idxmin(), 'title']
# average_runtime
average_runtime = data['runtime'].mean()
# max runtime
max_runtime = data['runtime'].max()
# min runtime
min_runtime = data['runtime'].min()

# max vote_count and title
most_vote_count_film = data.loc[data['vote_count'].idxmax(), 'title']
# min vote_count and title
most_vote_count_film = data.loc[data['vote_count'].idxmin(), 'title']
# average_vote_average
max_vote_count = data['vote_count'].max()
# min vote_average
min_vote_count = data['vote_count'].min()


# max vote_average and title
most_vote_average_film = data.loc[data['vote_average'].idxmax(), 'title']
# min vote_average and title
most_vote_average_film = data.loc[data['vote_average'].idxmin(), 'title']
# average_vote_average
max_vote_average = data['vote_average'].max()
# min vote_average
min_vote_average = data['vote_average'].min()

# max budget and title
most_budget_film = data.loc[data['budget'].idxmax(), 'title']
# min budget and title
most_budget_film = data.loc[data['budget'].idxmin(), 'title']
# average_budget
average_budget = data['budget'].mean()
# max budget
max_budget = data['budget'].max()
# min budget
min_budget = data['budget'].min()

# max revenue and title
most_revenue_film = data.loc[data['revenue'].idxmax(), 'title']
# min year and title
min_revenue_film = data.loc[data['revenue'].idxmin(), 'title']
# average_revenue
average_revenue = data['revenue'].mean()
# max revenue
max_revenue = data['revenue'].max()
# min revenue
min_revenue = data['revenue'].min()

# max year
data['year'] = pd.to_datetime(data['release_date']).dt.year
most_films_year = data['year'].value_counts().idxmax()
# min year
data['year'] = pd.to_datetime(data['release_date']).dt.year
min_films_year = data['year'].value_counts().idxmin()

# max original_language
max_original_language = data['original_language'].value_counts().idxmax()
# min original_language
min_original_language = data['original_language'].value_counts().idxmin()

# max_production_companies
max_production_companies = data['production_companies'].value_counts().idxmax()
# min_production_companies
min_production_companies = data['production_companies'].value_counts().idxmin()

# row with the highest revenue and lowest budget
result = data[(data['revenue'] > data['budget'])].count()
# row with the lowest revenue and highest budget
result = data[(data['revenue'] < data['budget'])].count()
# row with the highest budget and lowest revenue with title
titles_list = data.loc[data['revenue'] < data['budget'], 'title'].tolist()
# row with the highest revenue and lowest budget with title
titles_list = data.loc[data['revenue'] > data['budget'], 'title'].tolist()

# row with the runtime greater than or equal to 60 minutes
result1 = data[(data['runtime'] >= 60)].count()
# row with the runtime less than 60 minutes
result2 = data[(data['runtime'] < 60)].count()
# row with the runtime greater than or equal to 60 minutes with title and genres and production_companies
result1 = data[data['runtime'] >= 60][[
    'title', 'genres', 'production_companies']]
# row with the runtime less than 60 minutes with title and genres and production_companies
result2 = data[data['runtime'] < 60][[
    'title', 'genres', 'production_companies']]

# title and genres and production_companies for arabic films
result = data[data['original_language'] == 'Arabic'][[
    'title', 'budget', 'revenue', 'production_companies']]

# Calculate the correlation between budget and revenue
correlation = data['budget'].corr(data['revenue'])

# Count the occurrences of each language
language_counts = data['original_language'].value_counts()

# Calculate the correlation between budget and revenue
correlation = data['budget'].corr(data['revenue'])
correlation
if correlation > 0.7:
    correlation_strength = 'strong positive'
elif correlation < -0.7:
    correlation_strength = 'strong negative'
else:
    correlation_strength = 'weak'

# Calculate the variance revenue
variance = data['revenue'].var()

# Group the DataFrame by 'language' and calculate the sum of 'popularity' for each language
grouped = data.groupby('original_language')['popularity'].sum()

# Find the language with the highest popularity
most_popular_language = grouped.idxmax()

# Filter the DataFrame to include only rows with 'Horror' in the genres list
horror_films = data[data['genres'].apply(lambda x: 'Horror' in x)]

# Group the filtered DataFrame by production company and count the occurrences
company_counts = horror_films['production_companies'].value_counts()

# Get the company with the highest count
most_common_company = company_counts.idxmax()

genre_counts = data['genres'].value_counts()

# Retrieving the top 10 movies with the highest revenue
top_10_revenue = data.nlargest(10, 'revenue')[['title', 'revenue']]

# Calculating the correlation between movie popularity and revenue
correlation = data['popularity'].corr(data['revenue'])
#################################################################################################

# Selecting the top 5 genres by movie count
top_5_genres = data['genres'].value_counts().head(5)
# Creating a bar chart of top 5 genres
plt.bar(top_5_genres.index, top_5_genres.values)
plt.xlabel('Genre')
plt.ylabel('Movie Count')
plt.title('Top 5 Genres by Movie Count')
plt.xticks(rotation=45)
plt.show()

# Creating a scatter plot of revenue vs. budget
plt.scatter(data['budget'], data['revenue'])
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.title('Revenue vs. Budget')
plt.show()

# Creating a histogram of movie runtimes
plt.hist(data['runtime'], bins=30)
plt.xlabel('Runtime')
plt.ylabel('Count')
plt.title('Distribution of Movie Runtimes')
plt.show()

# Grouping the data by original language and calculating the average popularity
average_popularity_by_language = data.groupby('original_language')['popularity'].mean()
# Creating the bar chart
plt.bar(average_popularity_by_language.index, average_popularity_by_language.values)
plt.xlabel('Average Popularity')
plt.ylabel('Original Language')
plt.title('Average Movie Popularity by Language')
plt.xticks(rotation=45)
plt.show()


# Grouping the data by release year and calculating the average budget
average_budget_by_year = data.groupby('release_date')['budget'].mean()
# Creating the line plot
plt.plot(average_budget_by_year.index, average_budget_by_year.values)
plt.xlabel('Release Year')
plt.ylabel('Average Budget')
plt.title('Average Movie Budget Trend')
plt.show()




data['release_year'] = data['release_date'].dt.year
average_runtime = data.groupby('release_year')['runtime'].mean()
average_runtime.plot(kind='line', figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('Average Runtime')
plt.title('Average Runtime of Movies Over the Years')
plt.show()


# Create a pivot table to analyze the count of revenue based on budget and runtime
x = pd.pivot_table(index='budget', columns='genres', values='revenue', data=data, aggfunc='count')
# Create a heatmap to visualize the pivot table
sns.heatmap(x, annot=True, fmt='0.0f')
plt.title('Revenue Count by Budget and Runtime')
plt.show()
# Create a count plot to analyze the distribution of revenue values
sns.countplot(x="revenue", data=data)
plt.title('Distribution of Revenue')
plt.show()
# Calculate the correlation matrix of the numeric columns in the data
corr = data.corr(numeric_only=True)
# Create a heatmap to visualize the correlation matrix
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()
# Create a scatter plot to analyze the relationship between budget and revenue
plt.scatter(data["budget"], data["revenue"])
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title('Budget vs. Revenue')
plt.show()

##############################################################################

X=data['vote_average']
y=data['vote_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
X_train=np.array(X_train).reshape(-1,1)
lr =LinearRegression()
lr.fit(X_train,y_train)
c=lr.intercept_
m=lr.coef_
y_pred_train=m*X_train+c
print(y_pred_train)
y_pred_train1=lr.predict(X_train)
print(y_pred_train1)
plt.scatter(X_train,y_train)
plt.scatter(X_train, y_pred_train, color='red')
plt.xlabel('vote_average')
plt.ylabel('vote_count')

# split data into predictor and target variables
X = data['vote_average']
y = data['vote_count']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)

# reshape X_train to 2D array
X_train = np.array(X_train).reshape(-1, 1)

# create linear regression model and fit to training data
lr = LinearRegression()
lr.fit(X_train, y_train)


c=lr.intercept_
m=lr.coef_
# predict target variable using the model
y_pred_train =m * X_train + c
y_pred_test = m * np.array(X_test).reshape(-1, 1) + c

# calculate R-squared value for training and testing sets
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"R-squared value for training set: {r2_train:.4f}")
print(f"R-squared value for testing set: {r2_test:.4f}")


# Select the relevant features and target variable
X = data[['budget', 'revenue', 'popularity']]
y = data['revenue']  # Assuming 'success' is a binary variable indicating movie success

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select relevant features and target variable
features = ['budget', 'vote_average', 'popularity']
target = 'revenue'

# Define a revenue threshold to classify movies as successful or not
revenue_threshold = 100000000  # Adjust this threshold as per your requirement

# Convert revenue to binary labels based on the threshold
data['success'] = data[target] >= revenue_threshold

# Split the data into features (X) and target (y)
X = data[features]
y = data['success']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred_linreg = linreg.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred_linreg)
print("Linear Regression Mean Squared Error:", mse)
########################################################################################################
