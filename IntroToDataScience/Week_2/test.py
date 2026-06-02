import pandas as pd
import numpy as np


#Terminology
##1. DataFrame: A 2-dimensional labeled data structure with columns of potentially different types.
# It is similar to a spreadsheet or SQL table.
##2. Series: A one-dimensional labeled array capable of holding any data type. 
#It is similar to a single column in a DataFrame.
##3. Wrangling: The process of cleaning, transforming, and organizing raw data into a format suitable for analysis.
print ("******************OFFSET********************")
arr_1 = np.array([1, 2, 3, 4, 5])
arr_1_offset = [x-4 for x in arr_1]
print("Original Array:", arr_1) # Output: [1, 2, 3, 4, 5]
print("Offset Array (Minused by 4):", arr_1_offset) # Output: [-3, -2, -1, 0, 1]
print("\n")

print ("******************SIGN********************")
arr_2 = np.array([-3, -2, -1, 0, 1, 2, 3])
sign_arr = np.sign(arr_2)
print("Original Array:", arr_2) # Output: [-3 -2 -1  0  1  2  3]
print("Sign Array (Converts to -1, 0, 1):", sign_arr) # Output: [-1 -1 -1  0  1  1  1]
print("\n")

print ("******************ABSOLUTE VALUE********************")
arr_3 = np.array([-3, -2, -1, 0, 1, 2, 3])
abs_arr = np.abs(arr_3)
print("Original Array:", arr_3) # Output: [-3 -2 -1  0  1  2  3]
print("Absolute Value Array (Converts to positive values):", abs_arr) # Output: [3 2 1 0 1 2 3]
print("\n")

print("******************ROUNDING********************")
print("Rounding Array (Rounded to nearest integer):")
arr_4 = np.array([1.2, 2.5, 3.7, 4.0, 5.9])
rounded_arr = np.round(arr_4)
print("Original Array:", arr_4) # Output: [1.2 2.5 3.7 4.0  5.9]
print("Rounded Array (Rounded to nearest integer):", rounded_arr) # Output: [1. 2. 4. 4. 6.]

print("\n")
num = 3.14159
print("Original Number:", num) # Output: 3.14159
rounded_num = round(num, 2)
print("Rounded Number (Rounded to 2 decimal places):", rounded_num) # Output: 3.14
print("\n")





df = pd.DataFrame({
    'scores': [85, 90, 78, 92, 88],
})
print ("******************DATAFRAME WRANGLING********************")
print("Original DataFrame:\n", df) # Output: 
#    scores
# 0      85
# 1      90 .. so on
df['offset']= df['scores'] - 80
df['abs_offset'] = abs(df['offset'])
df['sign_offset'] = np.sign(df['offset'])
print("DataFrame with Offsets:\n", df) # Output:
#    scores  offset  abs_offset  sign_offset
# 0      85       5           5            1
# 1      90      10          10            1
# 2      78      -2           2           -1
# 3      92      12          12            1
# 4      88       8           8            1
print("\n")

df = df.rename(columns={'scores': 'original_scores', 'offset': 'score_offset', 'abs_offset': 'absolute_score_offset', 'sign_offset': 'sign_of_score_offset'})
print("Renamed DataFrame:\n", df) # Output:
#    original_scores  score_offset  absolute_score_offset  sign_of_score_offset
# 0               85             5                      5                    1
#...

df = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie', 'Jose', 'Dracula'],
    'age': ["20.45", "21.5", "19.99", "42.0", "300.0"],
})
print("New DataFrame - Students:\n", df) # Output:
#   student  age
# 0   Alice   20
# 1     Bob   21
#...
print("DataFrame Types:\n", df.dtypes) # Output:
# student    object
# age         object
# dtype: object
print("\n")

df["age"] = df["age"].astype(float)
print("DataFrame with Converted Age to Float:\n", df.dtypes) # Output:
# student    object
# age         float64
# dtype: object
print("\n")

df['categories'] = pd.cut(x=df['age'], bins=[0, 21, 50, 300], labels=['Young Adult', 'Adult', 'Vampire'])
print("DataFrame with Age Categories:\n", df) # Output:
#     student  age category
# 0     Alice   20.45  Young Adult
# 1       Bob   21.5  Adult
# 2   Charlie   19.99  Young Adult
# 3      Jose   42.0  Adult
# 4  Dracula   300.0  Vampire
print("\n")

df['age'] = df['age'].round()
df['categories'] = pd.cut(x=df['age'], bins=[0, 21, 50, 300], labels=['Young Adult', 'Adult', 'Vampire'])
print("DataFrame with Age Categories Rounded:\n", df) # Output:
#     student  age category
# 0     Alice   20.4  Young Adult
# 1       Bob   21.5  Adult
# 2   Charlie   20.0  Young Adult
# 3      Jose   42.0  Adult
# 4  Dracula   300.0  Vampire
print("\n")

df['categories_autobin'] = pd.cut(x=df['age'], bins = 3)
print("DataFrame with Auto Bins set to 3", df)
# 0    Alice   20.0  Young Adult  (18.719, 112.667]
# 1      Bob   22.0  Young Adult  (18.719, 112.667]
# 2  Charlie   20.0  Young Adult  (18.719, 112.667]
# 3     Jose   42.0        Adult  (18.719, 112.667]
# 4  Dracula  300.0      Vampire   (206.333, 300.0]
print("\n")

cross = pd.crosstab(index=df["categories"], columns="count")
print("DataFrame with Crosstab functionality", cross)
print("\n")

df = pd.DataFrame({
    'coworkers': ['Steve', 'Bob', 'Charlie'],
    'fulltime': ["yes", "yes", "no"],
})
print("New DataFrame - Coworkers:\n", df) # Output:
# 0     Steve       yes
# 1       Bob       yes
# 2   Charlie        no
print("\n")
#NOTE: .replace is to be deprecated. use .map instead.

df["fulltime"] = df["fulltime"].map({
    "yes": 1,
    "no": 0
    })
print("converted string to numerical", df)
print("\n")


print("\n")
df = pd.DataFrame({
    'name': [' Alice', 'JOHN', 'maria'],
    'city': ['New York', 'los angeles', 'Chicago']
})

print("New DataFrame - Names and Cities (For string manipulation):\n", df) # Output:
#      name         city
# 0   Alice     New York
# 1     JOHN  los angeles
# 2   maria      Chicago
print("\n")

df["city"] = df["city"].str.lower()
print("City names converted to lowercase:\n", df) # Output:
#      name         city
# 0   Alice     new york
# ...
print("\n")
df["city"] = df["city"].str.upper()
print("City names converted to uppercase:\n", df) # Output:
#      name         city
# 0   Alice     NEW YORK
# ...
print("\n")
df["name"] = df["name"].str.strip()
print("Names with whitespace stripped:\n", df) # Output:
#      name         city
# 0   Alice     NEW YORK
# 1     JOHN  LOS ANGELES
# ...
print("\n")

name_list = df["name"].tolist()
print("Names converted to list:\n", name_list) # Output: ['Alice', 'JOHN', 'maria']
print("\n")

df["contains_a"] = df["name"].str.contains("a", case=False) #Case=False makes it case insensitive
print("DataFrame with 'contains_a' column:\n", df) # Output:
#      name         city  contains_a
# 0   Alice     NEW YORK        True
# 1     JOHN  LOS ANGELES       False
# 2   maria      CHICAGO        True
print("\n")

print("******************SPLITTING AND CONCAT********************")

first_one_column = df.loc[:, "name"]
#Note, this is a Series because we are selecting only one column. If we wanted to keep it as a DataFrame, we would use df.loc[:, ["name"]]
print("First one column:\n", first_one_column)
print("\n")

last_two_columns = df.loc[:, "city":"contains_a"]
print("Last two columns:\n", last_two_columns)
print("\n")

df_joined = first_one_column.to_frame().join(last_two_columns) 
#NOTE: Because first_one_column is a Series as it is only one column, we need to convert it to a DataFrame using .to_frame() before joining.
print("Joined DataFrame:\n", df_joined) 
print("\n")

age=[17, 19, 21, 37, 45, 60]
score=[85, 90, 78, 92, 88, 95]
rt=[0.5, 0.7, 0.6, 0.8, 0.9, 1.0]
group=['test', 'control', 'test', 'control', 'test', 'control']

df = pd.DataFrame({
    'age': age,
    'score': score,
    'rt': rt,
    'group': group
})

print("Original DataFrame:\n", df) # Output:

df_test = df[df['group'] == 'test']
print("Filtered DataFrame for 'test' group:\n", df_test) # Output:
#    age  score   rt   group
# 0   17     85  0.5    test
# 2   21     78  0.6    test (only shows rows where group is 'test'
# ...)

df_control = df[df['group'] == 'control']
print("Filtered DataFrame for 'control' group:\n", df_control) # Output: Same as above but only shows rows where group is 'control'

df_concatenated = pd.concat([df_test, df_control], ignore_index=True)
print("Concatenated DataFrame:\n", df_concatenated) # Output: the original DataFrame but with the index reset to be sequential from 0 to n-1, where n is the total number of rows in the concatenated DataFrame.

print("\n")

print("******************MERGING********************")

students = pd.DataFrame({
    'studient_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David']
})

print("Students DataFrame:\n", students) # Output: You should know by now

scores = pd.DataFrame({
    'studient_id': [1, 2, 5],
    'score': [85, 90, 78]
})
print("Scores DataFrame:\n", scores) # Output: You should know by now

#This one will only merge the rows where the studient_id matches in both DataFrames, which are 1 and 2. The row with studient_id 3 from the students DataFrame and the row with studient_id 5 from the scores DataFrame will be excluded from the result.
inner_merge = pd.merge(students, scores, on='studient_id', how='inner')

print("Inner Merge DataFrame:\n", inner_merge) # Output:
#    studient_id     name  score
# 0            1    Alice     85
# 1            2      Bob     90
print("\n")

left_merge = pd.merge(students, scores, on='studient_id', how='left')
print("Left Merge DataFrame:\n", left_merge) # Output:
#    studient_id     name  score
# 0            1    Alice   85.0
# 1            2      Bob   90.0
# 2            3  Charlie    NaN
# 3            4    David    NaN
print("\n")

right_merge = pd.merge(students, scores, on='studient_id', how='right')
print("Right Merge DataFrame:\n", right_merge) # Output:
#    studient_id     name  score
# 0            1    Alice   85.0
# 1            2      Bob   90.0
# 2            5      NaN   78.0
print("\n")

outer_merge = pd.merge(students, scores, on='studient_id', how='outer')
print("Outer Merge DataFrame:\n", outer_merge) # Output:
#    studient_id     name  score
# 0            1    Alice   85.0
# 1            2      Bob   90.0
# 2            3  Charlie    NaN
# 3            4    David    NaN
# 4            5      NaN   78.0
print("\n")
print("\n")


drug_intake = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'alcohol': [3.7, 2.5, 4.0, 1.8, 3.2, 2.9, 4.5],
    'caffeine': [1.2, 0.8, 1.5, 0.5, 1.0, 0.9, 1.8],
    'none': [0.1, 0.2, 0.0, 0.3, 0.1, 0.4, 0.0]
})

print("Drug Intake DataFrame:\n", drug_intake) # Output: You should know by now

df_long = pd.melt(drug_intake, id_vars=['id', 'gender'])
print("Long Format DataFrame:\n", df_long) # Output:
#    id
# 0   1  M  alcohol    3.7
# 1   2  F  alcohol    2.5
# 2   3  M  alcohol    4.0
#...
# 7  1  M  caffeine   1.2
# 8  2  F  caffeine   0.8
#...
# 14 1  M  none       0.1
# 15 2  F  none       0.2
print("\n")

df_drop_duplicates = df_long.drop_duplicates(subset=['id'], keep='first') # Can only be first, last or False.
print("DataFrame after dropping duplicates:\n", df_drop_duplicates) # Output: Only the first occurrence of each id is kept
print("\n")

print("******************HANDLING MISSING VALUES********************")
df_with_missing = pd.DataFrame({
    'var1': [10, 20, float('nan'), 30],
    'var2': [10, float('nan'), float('nan'), 35],
    'var3': [float('nan'), 12, 18, 27]
})
print("\n")

print("DataFrame with Missing Values:\n", df_with_missing) # Output: You should know by now
check_for_missing = df_with_missing.isna()
print("Check for Missing Values (True indicates missing):\n", check_for_missing) 
#    var1   var2   var3
# 0  False  False   True
# 1  False   True  False
#...
print("\n")
print("Check sum of missing values in each column:\n", df_with_missing.isna().sum()) 
# Output: var1    1
# var2    2
# var3    1
#...
print("\n")

print("Drop rows with any missing values:\n", df_with_missing.dropna()) # Output: Only the row with no missing values is kept
print("\n")

print("Drop rows with missing values in var2:\n", df_with_missing.dropna(subset=['var2'])) # Output: Only the rows where var2 is not missing are kept
print("\n")

print("Fill missing values with 0:\n", df_with_missing.fillna(0)) # Output: All missing values are replaced with 0
print("\n")

print("******************DATA GROUPING********************")
df = pd.DataFrame({
    "department": ["HR", "HR", "IT", "IT", "Sales", "Sales"],
    "salary": [50000, 55000, 60000, 65000, 45000, 47000]
})
print("Original DataFrame:\n", df) # Output: You should know by now

grouped = df.groupby("department")["salary"].mean()
print("Average Salary by Department:\n", grouped) # Output:
# department
# HR       52500.0
# IT       62500.0
# Sales    46000.0
print("\n")

print("Group by size", df.groupby("department").size()) # Output:
# department
# HR       2
# IT       2
# Sales    2
print("\n")
