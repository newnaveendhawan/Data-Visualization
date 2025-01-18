#!/usr/bin/env python
# coding: utf-8

# ### (1) Setup
# ##### (1.1) Installing Pandas and packages
# ##### (1.2) Importing packages
# ### (2) Importing data
# ##### (2.1) Importing .csv files
# ##### (2.2) Other ways of creating DataFrames
# ##### (2.3) Changing names and data types
# ### (3) Summarizing data
# ##### (3.1) Peeking at the data
# ##### (3.2) Null values and summary statistics
# ##### (3.3) Unique values, value counts and sorting
# ##### (3.4) Basic visualizations
# ### (4) Selecting and computing new columns
# ##### (4.1) Accessing rows, columns and data
# ##### (4.2) Selecting subsets of columns
# ##### (4.3) Selecting subsets of rows
# ##### (4.4) Selecting subsets of rows and columns
# ##### (4.5) Creating new columns
# ##### (4.6) Applying functions

# In[ ]:





# #  Setup
# ### (1.1) Installing Pandas and package

# In[1]:


get_ipython().system('pip install pandas')


# ### (1.2) Importing packages

# In[2]:


import pandas as pd
import numpy as np

get_ipython().system(' python --version')


# In[3]:


pd. __version__


# ## Importing data
# #### Importing .csv files

# In[ ]:





# In[4]:


# df_csv = pd.read_csv('C:\Users\newna\Downloads\pokemon_first_50 - Sheet1.csv')


# In[ ]:





# In[5]:


df_csv = pd.read_csv(r'C:\Users\newna\Downloads\pokemon_first_50 - Sheet1.csv')
# This makes sure that the interpreter treats the file path as a raw string


# In[6]:


df_csv.head()


# In[7]:


df_csv.describe()


# In[8]:


# To avoid escape sequences: Python uses backslashes for special purposes, such as:

# \n for newlines
# \t for tabs
# \\ for a literal backslash
# 'C:\new_folder'  # \n is interpreted as a newline, so it breaks the path
# Using a raw string simplifies this:
#     r'C:\Users\newna\Downloads\pokemon.csv'


# In[9]:


example_string = "This is a string with special characters:\n\t- Newline and tab used here.\t- Path with a backslash: C:\\Users\\newna\\Documents\\file.txt"
print(example_string)


# In[10]:


df_csv.head()


# #### Importing .xlsx files

# In[11]:


# df_xlsx = pd.read_excel(r"C:\Users\newna\Downloads\pokemon_first_50.xlsx")


# In[12]:


# df_xlsx = pd.read_excel("C:\Users\newna\Downloads\pokemon_first_50.xlsx")


# In[13]:


get_ipython().system('pip install openpyxl')


# In[14]:


# pip uninstall openpyxl


# In[15]:


# pip install openpyxl --upgrade --pre


# In[16]:


df_xlsx1 = pd.read_excel("C:/Users/newna/Downloads/pokemon_first_50.xlsx")
df_xlsx1.head()


# In[17]:


df_xlsx1.head()


# In[18]:


#Ensure that you have the openpyxl library installed, as it's required 
#for reading .xlsx files. You can install it by running
# !pip install openpyxl


# In[19]:


df_csv.shape  # Alternatively, use len(df) for row count


# In[20]:


df_html = pd.read_html(r"C:\Users\newna\Downloads\pokemon_first_50\Sheet1.html")


# In[21]:


# df_html.head()


# In[22]:


df_html_list = pd.read_html(r"C:\Users\newna\Downloads\pokemon_first_50\Sheet1.html")
# # Select the first DataFrame from the list
df_html = df_html_list[0]

df_html.head()


# In[23]:


df_web1=pd.read_html("https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list")


# In[24]:


df_web = df_web1[0]


# In[25]:


print(df_web.head())


# In[26]:


df_web = df_web1[0]

# Alternatively, you can access by column name if you know it, for example:
first_column = df_web['Bank Name']

print(first_column.head())


# ### Other ways of creating DataFrames
# ##### Creating a DataFrame

# In[27]:


pd.DataFrame({'name':['Mahesh', 'Manoj', 'Manish'], 'age':[31, 25, 38]})


# In[28]:


dataframe=pd.DataFrame({'name':['Mahesh', 'Manoj', 'Manish'], 'age':[31, 25, 38]})
dataframe


# ## Changing names and data types

# In[29]:


url = r'https://en.wikipedia.org/wiki/List_of_Germans_by_net_worth'
tables = pd.read_html(url)
df_net_worth = tables[0]
df_net_worth.head()


# In[30]:


df_net_worth = (df_net_worth
                .rename(columns={'Net worth (USD)': 'net_worth',
                                'World ranking': 'world_ranking',
                                'Sources of wealth': 'wealth_source'}))

df_net_worth.rename(columns=str.capitalize).head(2)


# In[31]:


df_net_worth.dtypes


# In[32]:


df_net_worth['net_worth_in_billion'] = (df_net_worth['net_worth'].str.replace(' billion', '').apply(float))

df_net_worth.head()


# In[ ]:





# In[33]:


# shift + tab use for info


# 
# # Summarizing data

# #### Peeking at the data

# In[34]:


# use df.head, df.tail and df.sample


# ### Index

# In[35]:


survey = pd.DataFrame({'James': ['I liked it', 'It could use a bit more salt'], 'Emily': ['It is too sweet', 'Yum!']},
                     index = ['Product A', 'Product B'])
survey


# In[36]:


# Reset index
# Try playing around with 'drop' and 'inplace' and see what they do

survey.reset_index(drop = True, inplace = True)
survey


# #### Renaming columns

# In[37]:


test_scores = pd.DataFrame({'Student_ID': [154, 973, 645], 'Science': [50, 75, 31], 'Geography': [88, 100, 66],
                            'Math': [72, 86, 94]})
test_scores


# In[38]:


test_scores = test_scores.set_index('Student_ID')
test_scores


# In[39]:


test_scores.rename(columns = {'Geography': 'Physics', 'Science': 'Arts'}, inplace = True)
test_scores


# #### Dropping columns and rows

# In[40]:


# Drop the 'Math' column

test_scores.drop(columns = 'Math')


# In[41]:


# Drop row with student_ID 973
# We can make this more robust once we learn the 'loc' function in the coming weeks 
 
test_scores.drop(154)


# In[42]:


test_scores


# #### Adding columns and rows

# In[43]:


# Create a new column for history subject

test_scores['History'] = [79, 70, 67]
test_scores


# In[44]:


# Add more product reviews from James and Emily
# Recall our survey dataframe

survey


# In[45]:


# Create two more rows

df5 = pd.DataFrame({'James': ['Not good', 'Meh'], 'Emily': ['My grandma can cook better', 'Pretty average']})
df5


# In[46]:


# pip install pandas


# In[47]:


# Use the 'append' function

# survey = survey.concat(df5, ignore_index = True)
# survey
# survey = pd.concat([survey, df5], ignore_index=True)
# survey


# In[48]:


# pip install pandas==1.5.3


# In[49]:


# survey = df.append(df5, ignore_index=True)
# survey


# #### Index-based selection

# In[50]:


# First row and all columns

survey.iloc[0, :]


# In[51]:


# Fourth column that is the Name column and all rows
# Since starting index is 0 fourth column corresponds to index number 3

test_scores.iloc[:, 3]


# In[52]:


# First three rows and all columns

df_csv.iloc[[0, 1, 2], :]


# In[53]:


# Bottom five rows of the dataframe

df_csv.iloc[-5:, :]


# In[54]:


df_csv.tail()


# #### Label-based selection

# In[55]:


# First row of the Name column

df_csv.loc[44, 'species']


# In[56]:


# First 5 rows of the species, height and speed

df_csv.loc[:4, ['species', 'height', 'speed']]


# #### Conditional Selection

# In[57]:


# Rows with age 50

df_csv.loc[df_csv['height'] == 1.0, :]


# In[58]:


# Rows with height 1.0 AND are weight = 29.5
# This is a subset of the above dataframe by filtering out weight

df_csv.loc[(df_csv['height'] == 1.0) & (df_csv['weight'] == 29.5) ,:]


# In[59]:


# Rows with height 1.0 OR have speed greater than or equal to 60

df_csv.loc[(df_csv['height'] == 1.0) | (df_csv['speed'] >= 60), :] 


# In[60]:


# Rows with height 1.0 AND have speed greater than or equal to 60

df_csv.loc[(df_csv['height'] == 1.0) & (df_csv['speed'] >= 60), :]


# In[61]:


# All the rows with null cabin column

df_csv.loc[df_csv['weight'].isnull(), :]


# In[62]:


# All rows with C or Q in Embarked column

df_csv.loc[df_csv['type_1'].isin(['grass', "bug"]), :]


# In[63]:


# Describe function on numerical variable

df_csv['base_experience'].describe()


# In[64]:


# Info function

df_csv.info()


# #### SORT

# In[65]:


data= pd.read_csv(r"C:\Users\newna\Downloads\vgsalesGlobale.csv\vgsalesGlobale.csv")


# In[66]:


data


# In[67]:


data["Name"].sort_values(ascending=True)


# In[68]:


data.sort_values("Name")


# In[69]:


data.sort_values("Year")


# In[70]:


data.sort_values(["Year","Name"])


# #### MIssing Data

# In[71]:


s=pd.Series(["Sam",np.nan,"Tim","Kim"])
s


# In[72]:


s.isnull()


# In[73]:


s.notnull()


# In[74]:


s[3]=None
s.isnull()


# In[75]:


s.dropna()


# In[76]:


from numpy import nan as NA


# In[77]:


df=pd.DataFrame([[1,2,3],[4,NA,5],
                 [NA,NA,NA]])
df


# In[78]:


df.dropna()


# In[79]:


df.dropna(how="all")


# In[80]:


df


# In[81]:


df[1]


# In[82]:


df[1]=NA
df


# In[83]:


df.dropna(axis=1,how="all")


# In[84]:


# df.dropna(thresh=3)
# df


# In[85]:


df.fillna(0)


# In[86]:


df.fillna({0:15,1:25,2:35})


# In[87]:


df


# In[88]:


df.fillna(0,inplace=True)
df


# In[89]:


# df


# In[90]:


df=pd.DataFrame([[1,2,3],[4,NA,5],
                 [NA,NA,NA]])
df


# In[91]:


df.fillna(method="ffill")
# method="ffill": This stands for "forward fill." 
#     It takes the last valid value encountered and fills 
#     any missing values (NaN) in the subsequent rows with this value.


# In[92]:


data=pd.Series([1,0,NA,5])
data


# In[93]:


data.fillna(data.mean())


# In[94]:


df


# In[95]:


df.fillna(df.mean()),


# # Mean Mode Mediun

# In[96]:


# Mean - The average value
# Median - The mid point value
# Mode - The most common value


# #### Mean (Average)

# In[97]:


# Returns the mean (average) of the data.


# In[98]:


from statistics import mean

data = [1, 2, 3, 4, 5]
result = mean(data)
print(result)
# print("Mean:", result)


# #### Median

# In[99]:


# Returns the median of the data.


# In[100]:


from statistics import median

data = [1, 2, 3, 4, 5]
result = median(data)
print("Median:", result)


# #### Mode

# In[101]:


# Returns the mode of the data (most common value)


# In[102]:


from statistics import mode

data = [1, 2, 2, 3, 3, 3, 4]
result = mode(data)
print("Mode:", result)


# #### Standard Deviation

# In[103]:


# Returns the standard deviation of the data


# In[104]:


from statistics import stdev

data = [1, 2, 3, 4, 5]
result = stdev(data)
print("Standard Deviation:", result)


# #### Variance

# In[105]:


# Returns the variance of the data


# In[106]:


from statistics import variance

data = [1, 2, 3, 4, 5]
result = variance(data)
print("Variance:", result)


# In[107]:


# by-numpy


# In[108]:


speed = [12,20,3,13,12,10,11,6,9,8,2,18,11,9,12] 
# 15 numbers


# In[109]:


(12+20+3+13+12+10+11+6+9+8+2+18+11+9+12)/15


# In[110]:


# Use the NumPy mean() method to find the average:


# In[111]:


x = np.mean(speed)
x


# In[112]:


x = np.median(speed)
x


# In[113]:


# x = np.mode(speed)
# x


# In[114]:


from scipy import stats
x = stats.mode(speed)
x


# In[115]:


x = np.std(speed)
x


# In[116]:


# Variance


# In[117]:


# Find the mean


# In[118]:


(32+111+138+28+59+77+97)/7


# In[119]:


# # find the difference from the mean
# 32 - 77.4 = -45.4
# 111 - 77.4 =  33.6
# 138 - 77.4 =  60.6
#  28 - 77.4 = -49.4
#  59 - 77.4 = -18.4
#  77 - 77.4 = - 0.4
#  97 - 77.4 =  19.6


# In[120]:


# For each difference: find the square value
# (-45.4)2 = 2061.16
#  (33.6)2 = 1128.96
#  (60.6)2 = 3672.36
# (-49.4)2 = 2440.36
# (-18.4)2 =  338.56
# (- 0.4)2 =    0.16
#  (19.6)2 =  384.16


# In[121]:


# The variance is the average number of these squared differences


# In[122]:


# (2061.16+1128.96+3672.36+2440.36+338.56+0.16+384.16) / 7 = 1432.2


# In[123]:


speed = [32,111,138,28,59,77,97]
x = np.var(speed)
x


# In[124]:


# Standard Deviation √1432.25 = 37.85


# In[125]:


import numpy
speed = [32,111,138,28,59,77,97]
x = numpy.std(speed)
print(x)


# In[126]:


df6 = pd.DataFrame({'player': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                   'game1': [18, 22, 19, 14, 14, 11, 20, 28],
                   'game2': [5, 7, 7, 9, 12, 9, 9, 4],
                   'game3': [11, 8, 10, 6, 6, 5, 9, 12],
                   'game4': [9, 8, 10, 9, 14, 15, 10, 11]})
print(df6)


# In[127]:


print(df6.mean(numeric_only=True))
print(df6.median(numeric_only=True))
print(df6.mode(numeric_only=True))


# In[128]:


data = [[1, 1, 2], [6, 4, 2], [3, 2, 1], [4, 2, 3]]
df7 = pd.DataFrame(data)
print(df7)
print(df7.median())
df7


# In[129]:


data = [[1, 1, 2], [6, 4, 2], [4, 2, 1], [4, 2, 3]]
df8 = pd.DataFrame(data)
print(df8.mean())
df8


# In[130]:


data = [[1, 1, 2], [6, 4, 2], [4, 2, 1], [4, 2, 3]]
df9 = pd.DataFrame(data)
print(df9.mode())
df9


# ## How to Convert Categorical Data to Numerical Data?

# In[131]:


# Integer Encoding


# In[132]:


# As a first step, each unique category value is assigned an integer value.

# For example, “red” is 1, “green” is 2, and “blue” is 3.


# In[133]:


df12 = pd.DataFrame({'Name':['John Smith', 'Mary Brown'],
 'Gender':['M', 'F'], 'Smoker':['Y', 'N']})

df12


# In[134]:


df_with_dummies = pd.get_dummies(df12, columns=['Gender', 'Smoker'])

df_with_dummies


# ## Dealing with Duplicate data

# In[135]:


df13 = pd.DataFrame({'A': [1, 2, 2, 3, 4, 4, 4],
   'B': [5, 6, 7, 8, 9, 10, 11]})

df13


# In[136]:


mask = df13.A.duplicated(keep=False)
mask


# In[137]:


df13.loc[mask,'B'] = 0


# In[138]:


df13['C'] = df13.A.mask(mask,0)
df13


# In[139]:


# df.loc[mask,'B'] = 0
# mask is a boolean Series or condition that specifies which rows should be updated.
# df.loc[mask, 'B'] selects all rows where mask is True and targets the column 'B'
# = 0 sets the values of column 'B' to 0 for the rows where mask is True


# d invert mask use ~:

# In[140]:


df13['C'] = df13.A.mask(~mask, 0)

df13


# ### Drop duplicated
pd.drop_duplicate
# In[ ]:





# In[141]:


df14 = pd.DataFrame({'A':[1,2,3,3,2],
'B':[1,7,3,0,8]})
df14


# In[142]:


# keep only the last value
df14.drop_duplicates(subset=['A'])


# In[143]:


# keep only the last value
df14.drop_duplicates(subset=['A'], keep='last')


# In[144]:


# Drop all duplicates
df_none = df14.drop_duplicates(subset=['A'], keep=False)

print(df_none)


# In[145]:


# Drop duplicates, keeping only the first occurrence
df_first = df14.drop_duplicates(subset=['A'], keep='first')

df_first


# In[146]:


# When you don't want to get a copy of a data frame, but to modify the existing one:


# In[147]:


df15 = pd.DataFrame({'A':[1,2,3,3,2],
 'B':[1,7,3,0,8]})
df15


# In[148]:


df15.drop_duplicates(subset=['A'], inplace=True)

df15


# #### Counting and getting unique elements

# In[149]:


# Number of unique elements in a series:


# In[150]:


id_numbers = pd.Series([111, 112, 112, 114, 115, 118, 114, 118, 112])

id_numbers.nunique()
# unique() It returns the unique values of a Series or DataFrame column.
# nunique() It returns the number of unique values in a Series or DataFrame column. (count of the unique elements)


# In[151]:


# Get unique elements in a series


# In[152]:


id_numbers.unique()


# ##### Number of unique elements in each group:

# In[153]:


df16 = pd.DataFrame({'Group': list('ABAABABAAB'),
 'ID': [1, 1, 2, 3, 3, 2, 1, 2, 1, 3]})

df16


# In[154]:


df16.groupby('Group')['ID'].nunique()
#  how (later)


# In[155]:


# Get of unique elements in each group:
df16.groupby('Group')['ID'].unique()


# ###### Get unique values from a column

# In[156]:


df17 = pd.DataFrame({"A":[1,1,2,3,1,1],"B":[5,4,3,4,6,7]})

df17["A"].unique()


# In[157]:


df17["B"].unique()


# In[158]:


# To get the unique values in column A as a list
pd.unique(df17['A']).tolist()


# ### Groupby Method
Group by one column
# In[159]:


import pandas as pd

df18 = pd.DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'b'],
 'B': [2, 8, 1, 4, 3, 8],
 'C': [102, 98, 107, 104, 115, 87]})

df18

Group by column A and get the mean value of other columns:
# In[160]:


df18.groupby('A').mean()

Group by multiple columns
# In[161]:


df18.groupby(['A','B']).mean()


# In[162]:


df20 = pd.DataFrame(
 {"Name":["Alice", "Bob", "Mallory", "Mallory", "Bob" , "Mallory"],
 "City":["corn", "corn", "cap", "corn", "corn", "cap"],
 "Val": [4, 3, 3, np.nan, np.nan, 4]})
df20


# In[163]:


df20.groupby(["Name", "City"])['Val'].size().reset_index(name='Size')


# In[164]:


df20


# #### Handling missing values

# In[165]:


# Adding a NaN value for demonstration
df20.at[2, 'Val'] = None
df20


# In[166]:


# Fill missing values
df20['Val'].fillna(df20['Val'].mean(),inplace=True)
df20


# In[167]:


df20['Val'] = df20['Val'].round(3)


# In[168]:


df20


# In[ ]:





# In[ ]:





# In[ ]:





# In[169]:


data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 65000, 120000, 95000],
    'Experience': [2, 5, 1, 10, 7]
}


# In[170]:


data = pd.DataFrame(data)
data


# In[171]:


print("Data types of columns:\n", data.dtypes)

Converting data types
# In[172]:



data['Age'] = data['Age'].astype(float)
data.dtypes


# In[173]:


# pivot and lambda function

Using multi-indexing
# In[174]:


data.set_index(['City', 'Name'], inplace=True)
data


# In[175]:


data.reset_index(inplace=True)
data


# In[176]:


# date time and arg agregate


# In[177]:


# Query functions


# In[178]:


# !pip install skimpy


# In[ ]:





# In[8]:


import pandas as pd

# Creating a DataFrame
data = {
    'Date': ['2024-05-01', '2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02', '2024-05-03', '2024-05-03', '2024-05-03'],
    'Item': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana', 'Orange', 'Apple', 'Orange'],
    'Units Sold': [30, 21, 15, 40, 34, 20, 45, 25],
    'Price Per Unit': [1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 0.75],
    'Salesperson': ['John', 'John', 'John', 'Alice', 'Alice', 'John', 'Alice', 'John']
}

# Filtering the Banana item and checking its data type
banana_rows = df22[df22['Item'] == 'Banana']
print(banana_rows['Item'].dtype)
Salesperson_rows = df22[df22['Salesperson'] == 'John']
print(Salesperson_rows['Salesperson'].dtype)
# Display the DataFrame
df22


# In[180]:


from skimpy import skim

skim(df22)


# In[181]:


# !pip install missingno


# In[182]:


# Import missingno library
import missingno

# Visualise null values
missingno.matrix(df22)


# In[183]:


df22.describe()


# In[184]:


df23=pd.Series(np.random.randn(8),
               index=[["a","a","a","b",
                       "b","b","c","c"],
                      [1,2,3,1,2,3,1,2]])
df23


# In[185]:


df23.unstack()


# In[186]:


df23.unstack().stack()


# In[187]:


df23=pd.DataFrame(
    np.arange(12).reshape(4,3),
    index=[["a","a","b","b"],
           [1,2,1,2]],
    columns=[["num","num","ver"],
             ["math","stat","geo"]])
df23


# In[188]:


df23.index.names=["class","exam"]
df23.columns.names=["field","lesson"]
df23


# In[189]:


df23["num"]


# In[190]:


df23.swaplevel("class","exam")

Summary Statistics by Level
# In[191]:


df


# In[192]:


# df23.sum(level="exam")
# df23.groupby(level="exam").sum()


# In[193]:


# df23.sum(level="field",axis=1)


# ### Combining & Merging Datasets in Pandas (EXTRA)
Joining DataFrame
# In[194]:


d1=pd.DataFrame(
    {"key":["a","b","c","c","d","e"],
     "num1":range(6)})
d2=pd.DataFrame(
    {"key":["b","c","e","f"],
     "num2":range(4)})


# In[195]:


print(d1)
print(d2)


# In[196]:


pd.merge(d1, d2)
# both


# In[197]:


pd.merge(d1, d2, on='key') # on function 
# Rows from d1 and d2 will be matched where the 'key' values are the same.


# In[198]:


d3=pd.DataFrame(
    {"key1":["a","b","c","c","d","e"],
     "num1":range(6)})
d4=pd.DataFrame(
    {"key2":["b","c","e","f"],
     "num2":range(4)})


# In[199]:


d3


# In[200]:


d4


# In[201]:


pd.merge(
    d3,d4,left_on="key1",right_on="key2"
)
# both common


# In[202]:


d1


# In[203]:


d2


# In[ ]:





# ![1m55Wqo.jpg](attachment:1m55Wqo.jpg)

# In[ ]:





# ![1_av8Om3HpG1MC7YTLKvyftg.webp](attachment:1_av8Om3HpG1MC7YTLKvyftg.webp)

# In[ ]:





# In[204]:


pd.merge(d1,d2,how="outer") #how (questions)
# all 


# In[205]:


d1


# In[206]:


d2


# In[207]:


pd.merge(d1,d2,how="left")


# In[208]:


pd.merge(d1,d2,how="right")


# In[209]:


pd.merge(d1, d2, how='inner')


# In[210]:


df1z=pd.DataFrame(
    {"key":["a","b","c","c","d","e"],
     "num1":range(6),
     "count":["one","three","two",
              "one","one","two"]})
df2z=pd.DataFrame(
    {"key":["b","c","e","f"],
     "num2":range(4),
     "count":["one","two","two","two"]})


# In[211]:


pd.merge(df1z, df2z, on=['key', 'count'], 
         how='outer')


# In[212]:


pd.merge(df1z, df2z, on="key", how='outer')


# In[213]:


pd.merge(df1z, df2z, 
         on='key', 
         suffixes=('_data1', '_data2'))

Merging on index
# In[214]:


df1a=pd.DataFrame(
    {"letter":["a","a","b",
               "b","a","c"],
     "num":range(6)}) 
df2a=pd.DataFrame(
    {"value":[3,5,7]},
    index=["a","b","e"])
df1a


# In[215]:


df2a


# In[216]:


print(df1a)
print(df2a)


# In[217]:


pd.merge(df1a,df2a,
         left_on="letter",
         right_index=True)


# In[218]:


#later join
right=pd.DataFrame(
    [[1,2],[3,4],[5,6]],
    index=["a","c","d"],
    columns=["Tom","Tim"])
left=pd.DataFrame(
    [[7,8],[9,10],[11,12],[13,14]],
    index=["a","b","e","f"],
    columns=["Sam","Kim"])


# In[219]:


right


# In[220]:


left


# In[221]:


pd.merge(right,left, 
         right_index=True, 
         left_index=True, 
         how="outer")


# In[222]:


left.join(right)


# In[223]:


left.join(right,how="outer")


# In[224]:


data=pd.DataFrame([[1,3],[5,7],[9,11]],            
                  index=["a","b","f"],      
                  columns=["Alex","Keta"])
left.join([right,data])

Concatenating Along an Axis #tomarrow
# In[225]:


seq= np.arange(20).reshape((4, 5))
seq


# In[226]:


np.concatenate([seq,seq], axis=1)


# In[227]:


np.concatenate([seq, seq], axis=0)


# In[228]:


data1 = pd.Series(
    [0, 1], index=['a', 'b'])
data2 = pd.Series(
    [2,3,4], index=['c','d','e'])
data3 = pd.Series(
    [5, 6], index=['f', 'g'])


# In[229]:


data1


# In[230]:


data2


# In[231]:


data3


# In[232]:


pd.concat([data1,data2,data3])


# In[233]:


pd.concat([data1, data2, data3], axis=1)


# In[234]:


data4= pd.Series([10,11,12], 
                 index=['a','b',"c"])
pd.concat([data1,data4],axis=1,join="inner")


# In[235]:


x=pd.concat([data1, data2, data4], 
            keys=['one', 'two','three'])
x


# In[236]:


x=pd.concat([data1, data2, data4], 
            axis=1,#column
            keys=['one', 'two', 'three'])
x


# In[237]:


# df1w = pd.DataFrame(
#     np.arange(6).reshape(3, 2),
#     index=['a', 'b', 'c'],
#     columns=['one', 'two'])
# df2w = pd.DataFrame(
#     10+np.arange(4).reshape(2,2),
#     index=['c', 'a'],
#     columns=['four','three '])
# print(df1w)
# print(df2w)
df1w = pd.DataFrame(
    np.arange(6).reshape(3, 2),
    index=['b', 'a', 'c'],  # Unsorted indices
    columns=['one', 'two'])

df2w = pd.DataFrame(
    10 + np.arange(4).reshape(2, 2),
    index=['c', 'a'],  # Partially sorted indices
    columns=['four', 'three'])


# In[238]:


print(df1w)


# In[239]:


print(df2w)


# In[240]:


pd.concat([df1w, df2w], axis=1, 
          keys=['s1', 's2'],
          sort=False)


# In[241]:


pd.concat([df1w, df2w], axis=1, 
          keys=['s1', 's2'],
          sort=True)


# In[242]:


data1 = pd.DataFrame(
    np.random.randn(3, 4),
    columns=['a','b','c','d'])
data2 = pd.DataFrame(
    np.random.randn(2, 3),
    columns=['b','d','a'])


# In[243]:


data1


# In[244]:


data2


# In[ ]:





# In[245]:


pd.concat([data1, data2], ignore_index=True)


# In[246]:


pd.concat([data1, data2], ignore_index=False)


# In[247]:


data=pd.DataFrame(
    np.arange(16).reshape(4,4),
    index=[list("aabb"),[1,2]*2],
    columns=[["num","num",
              "comp","comp"],
             ["math","stat"]*2])
data


# In[248]:


data.index.names=["class","exam"]
data.columns.names=["field","lesson"]
data


# In[249]:


long=data.stack()


# In[250]:


# long


# In[251]:


# long.unstack()


# In[252]:


# data.stack()


# In[253]:


data.stack(0)


# In[254]:


data.stack("field")


# In[255]:


s1=pd.Series(
    np.arange(4),index=list("abcd"))
s2=pd.Series(
    np.arange(6,9),index=list("cde"))


# In[256]:


print(s1)
print(s2)


# In[257]:


data2=pd.concat([s1,s2],keys=["c1","c2"])
data2


# In[258]:


data2.unstack()


# In[259]:


data2.unstack().stack(dropna=False)


# In[260]:


data2.unstack().stack(dropna=False)

Pivoting “Long” to “Wide” Format (later)
# In[261]:


stock=pd.DataFrame(
    {"fruit": ["apple", "plum","grape"]*2,
     "color": ["purple","yellow"]*3,
     "piece":[3,4,5,6,1,2]})


# In[262]:


stock


# In[263]:


# stock.pivot(index=None, columns=None, values=None)


# In[264]:


stock["value"]=np.random.randn(len(stock))


# In[265]:


stock


# In[266]:


df_copy = stock.copy()


# In[267]:


pip install --upgrade pandas


# In[268]:


pip install --upgrade pandas --pre


# In[269]:


import pandas as pd
print(pd.__version__)


# In[270]:


import pandas as pd


# In[271]:


dfw = pd.DataFrame({'fruit': ['apple', 'banana', 'orange', 'apple'],
                    'color': ['red', 'yellow', 'orange', 'green']})
df_copy = dfw.copy()


# In[272]:


# p = df_copy.pivot("fruit", "color")


# In[ ]:





# In[ ]:





# In[273]:


# p=df_copy.pivot("fruit","color")
# p


# In[274]:


# p["value"]

Pivoting “Wide” to “Long” Format
# In[275]:


data=pd.DataFrame(
    {"lesson":["math","stat","bio"],
     "Sam":[50,60,70],
     "Kim":[80,70,90],
     "Tom":[60,70,85]})
data


# In[276]:


# group=pd.melt(data,["lesson"])


# In[278]:


# group


# In[279]:


# data=group.pivot(
#     "lesson","variable","value")
# data


# In[280]:


data.reset_index()


# ## What is Groupby in Pandas? (EXTRA)

# In[281]:


dfkey=pd.DataFrame(
    {"key1":list("aabbab"),
     "key2":["one","two","three"]*2,
     "data1":np.random.randn(6),
     "data2":np.random.randn(6)})
dfkey


# In[ ]:





# In[282]:


print(type(dfkey))


# In[283]:


list(dfkey)


# In[284]:


df


# In[285]:


type(df)


# In[286]:



group=dfkey["data1"].groupby(dfkey["key1"])


# In[287]:


group


# In[288]:


group.mean()


# In[289]:


ave=dfkey["data1"].groupby([dfkey["key1"],
                         dfkey["key2"]]).mean()
ave


# In[290]:


ave.unstack()


# In[291]:


# dfkey.groupby("key1").mean()


# In[292]:


dfkey.groupby(["key1","key2"]).mean() #do youself


# In[293]:


# Iterating over Groups


# In[294]:


g=dfkey.groupby("key1")
g


# In[295]:


for name, group in dfkey.groupby("key1"):
    print(name)
    print(group)
    
#do yourself


# In[296]:


for (x1,x2),group in dfkey.groupby(["key1",
                                 "key2"]):
    print(x1,x2)
    print(group)


# In[297]:


dfkey.groupby("key1")


# In[298]:


piece=dict(list(dfkey.groupby("key1")))
piece


# In[299]:


piece["a"]


# In[300]:


# Selecting a Column or Subset of Columns


# In[301]:


dfkey


# In[302]:


dfkey.groupby(['key1', 
            'key2'])[['data1']].mean()


# In[303]:


# Grouping with Dicts and Series


# In[304]:


fruit=pd.DataFrame(np.random.randn(4,4),
                   columns=list("abcd"),
                   index=["apple","cherry",
                          "banana","kiwi"])
fruit


# In[305]:


label={"a": "green","b":"yellow",
       "c":"green","d":"yellow",
       "e":"purple"}


# In[306]:


group=fruit.groupby(label,axis=1)
# groupby(label, axis=1) groups the columns based on the values in the label dictionary.


# In[307]:


group.sum()
# The output shows the sum of 'green' (columns 'a' and 'c') and 'yellow' (columns 'b' and 'd') for each fruit.
# kiwi 1.33(b) + 0.33(d)


# In[308]:


s=pd.Series(label)
s


# In[309]:


fruit.groupby(s,axis=1).count()


# In[310]:


# Grouping with Functions


# In[311]:


fruit.groupby(len).sum()


# In[ ]:





# ### Working with Text Data (EXTRA FAST)

# In[312]:


"hello".upper()


# In[313]:


# Vectorized String Functions


# In[314]:


data=["tim","Kate","SUSan",np.nan,"aLEX"]


# In[315]:


name=pd.Series(data)
name


# In[316]:


p = 'intr'
print(name.str.capitalize())
p.capitalize()


# In[317]:


name.str.lower()


# In[318]:


name.str.len()


# In[319]:


name.str.startswith("S")


# In[320]:


dataframe1=pd.DataFrame(
    np.random.randn(3,2),
    columns=["Column A","Column B"],
    index=range(3))
dataframe1


# In[321]:


dataframe1.columns


# In[322]:


dataframe1.columns.str.lower().str.replace(" ","_")


# In[323]:


s=pd.Series(["a_b_c","c_d_e",np.nan,"f_gh"])
s


# In[324]:


a=s.str.split("_")
a


# In[325]:



a.str[1]


# In[ ]:





# In[326]:



s.str.split("_",expand=True,n=2)


# In[327]:


money=pd.Series(["15","-$20","$30000"])
money


# In[328]:


money.str.replace("-$","")


# In[341]:


# Use str.replace to remove $ and - signs
money.str.replace("[$-]", "", regex=True)
# "[$-]" is a regular expression that matches both the $ and - symbols
# regex=True is used to indicate that we are passing a regular expression for the replace function


# In[330]:


# pivot, lambda, argregate, groupby high,


# In[360]:


dataframe1


# In[361]:


dataframe1.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




