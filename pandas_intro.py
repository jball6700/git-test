#Imports
import numpy as np
import pandas as pd


#Object Creation

#Create a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns:
dates = pd.date_range("20130101", periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index= dates, columns=list("ABCD"))
print(df)

#Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df2 = pd.DataFrame(
{
"A": 1.0,
"B": pd.Timestamp("20130102"),
"C": pd.Series(1, index=list(range(4)), dtype="float32"),
"D": np.array([3] * 4, dtype="int32"),
"E": pd.Categorical(["test", "train", "test", "train"]),
"F": "foo",
}
)
print(df2)

#Collums of DataFrame have different dtypes
print(df2.dtypes)


#Viewing Data

#View top and bottom rows of the frame
print(df.head())
print(df.tail(3))

#Display Index/Collumns
print(df.index)
print(df.columns)

#For df, our DataFrame of all floating-point values, converting to numpy, doesn’t require copying data.
print(df.to_numpy())

#For df2, the DataFrame with multiple dtypes, converting to numpy is relatively expensive
print(df2.to_numpy())

#describe() shows a quick statistic summary of your data:
print(df.describe())

#Transposing your data
print(df.T)

#Sorting by axis
print(df.sort_index(axis=1, ascending=False))

#Sorting by values
print(df.sort_values(by="B"))


#Selection

#Getting
#Selecting a single column, which yields a Series, equivalent to df.A:
print(df["A"])

#Selecting via [], which slices the rows.
print(df[0:3])

print(df["20130102":"20130104"])


#Selection by label
#For getting a cross section using a label:
print(df.loc[dates[0]])

#Selecting on a multi-axis by label:
print(df.loc[:, ["A","B"]])

#showing label slicing, both endpoints are included:
print(df.loc["20130102":"20130104", ["A" , "B"]])

#Reduction in the dimensions of the returned object:
print(df.loc["20130102", ["A", "B"]])

#For getting a scaler value
print(df.loc[dates[0], "A"])
print(df.at[dates[0], "A"])


#Selection by position
#Select via the position of the passed integers:
print(df.iloc[3])

#By integer slices, acting similar to numpy/Python:
print(df.iloc[3:5, 0:2])

#By lists of integer position locations, similar to the NumPy/Python style:
print(df.iloc[[1, 2, 4], [0, 2]])

#For slicing rows:
print(df.iloc[1:3, :])

#For slicing columns:
print(df.iloc[:, 1:3])

#For getting a value explicitly:
print(df.iat[1, 1])


#Boolean indexing
#Using a single column’s values to select data.
print(df[df["A"] > 0])

#Selecting values from a DataFrame where a boolean condition is met.
print(df[df > 0])

#Using isin() method for filtering:
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print(df2)
print(df2[df2["E"].isin(["two", "four"])])


#Setting
#Setting a new column automatically aligns the data by the indexes.
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
print(s1)
df["F"] = s1

#Setting values by label
df.at[dates[0], "A"] = 0

#Setting values by position:
df.iat[0, 1] = 0

#Setting by assigning with a NumPy array
df.loc[:, "D"] = np.array([5] * len(df))

#Result of change in settings
print(df)

#Where operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)


#Missing Data

#Reindexing allows you to change/add/delete the index on a specified axis. Returns a copy
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
print(df1)

#Drop rows with missing data
print(df1.dropna(how="any"))

#Filling missing Data
print(df1.fillna(value=5))

#Boolean values for when data are nan
print(pd.isna(df1))


#Operations

#Stats
#Performing descriptive statistic
print(df.mean())

#Same opperation on other axis
print(df.mean(1))

#Operating with objects that have different dimensionality that need alignment.
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)
print(df.sub(s, axis="index"))

#Apply
#Applying functions to the data
print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))

#Histograming
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)

print(s.value_counts())

#String Methods
#Series is equipped with a set of string processing methods in the str attribute that make it easy to operate on each element of the array
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print(s)

print(s.str.lower())


#Merge

#Concat
#Concatenating pandas objects together with concat():
df = pd.DataFrame(np.random.randn(10, 4))
print(df)

pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))

#Join
#SQL style merges.
left = pd.DataFrame({"key": ["foo", "bar"], "leftval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rightval": [4, 5]})
print(left)
print(right)

print(pd.merge(left, right, on="key"))


#Grouping
#Create DataFrame
df = pd.DataFrame(
{
"A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
"B": ["one", "one", "two", "three", "two", "two", "one", "three"],
"C": np.random.randn(8),
"D": np.random.randn(8),
}
)

print(df)

#Grouping and then applying the sum() function to the resulting groups.
print(df.groupby("A").sum())

#Grouping by multiple columns then applying sum function
print(df.groupby(["A", "B"]).sum())
