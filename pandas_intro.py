
#Imports
import numpy as np
import pandas as pd

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

#View top and bottom rows of the frame
print(df.head())
print(df.tail(3))

#Display Index/Collumns
print(df.index)
print(df.columns)

# For df, our DataFrame of all floating-point values, converting to numpy, doesn’t require copying data.
print(df.to_numpy())

# For df2, the DataFrame with multiple dtypes, converting to numpy is relatively expensive
print()
