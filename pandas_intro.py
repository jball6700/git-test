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


#Reshaping
#stack
tuples = list(
    zip(
        *[
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
    )
)

index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
print(df2)

#Pivot Tables
df = pd.DataFrame(
    {
    "A": ["one", "one", "two", "three"]*3,
    "B": ["A","B", "C"]*4,
    "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
    "D": np.random.randn(12),
    "E": np.random.randn(12),
    }
)

print(df)

print(pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]))


#Time series
#performing resampling operations during frequency conversion
rng = pd.date_range("1/1/2012", periods=100, freq="S")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample("5Min").sum())

#Timezone representation
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

#Converting to another time zone:
print(ts_utc.tz_convert("US/Eastern"))

#Converting between time span representations
rng = pd.date_range("1/1/2012", periods=5, freq="M")
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)

ps = ts.to_period()
print(ps)
print(ps.to_timestamp())

#convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end.
prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9
print(ts.head())


#Categoricals
#pandas can include categorical data in a DataFrame
df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)

df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])

#Rename the categories to more meaningful names
df["grade"].cat.categories = ["very good", "good", "very bad"]

#Reorder the categories and simultaneously add the missing categories
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
print(df["grade"])

#Sorting is per order in categories, not lexical order!
print(df.sort_values(by="grade"))

#Grouping by a categorical column also shows empty categories.
print(df.groupby("grade").size())


#Plotting

#use the standard convention for referencing the matplotlib API:
import matplotlib.pyplot as plt
plt.close("all")
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
print(ts.plot())

#On a DataFrame, the plot() method is a convenience to plot all of the columns with labels:
df = pd.DataFrame(
        np.random.randn(1000, 4), index=ts.index, columns=["A","B","C","D"]
    )
df = df.cumsum()
print(plt.figure())
print(df.plot())



#Getting data in/out

#CSV
#Writing to a csv file
df.to_csv("foo.csv")

#reading from a csv file
print(pd.read_csv("foo.csv"))
