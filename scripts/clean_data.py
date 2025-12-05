import pandas as pd
import re

df = pd.read_csv('/Users/seb/Desktop/Project2/nsw_property_data.csv', low_memory=False)

# Convert to date type
df['Contract date'] = pd.to_datetime(df['Contract date'], errors='coerce')
df['Settlement date'] = pd.to_datetime(df['Settlement date'], errors='coerce')

# Convert to int
df['Property post code'] = df['Property post code'].astype('Int64')
df['Purchase price'] = df['Purchase price'].astype('Int64')

df = df[df['Purchase price'].notna()]
df = df[df['Purchase price'] > 0]
df = df[df['Property post code'].notna()]

# Standardise the street names
df['Property street name'] = (
    df['Property street name']
    .str.upper()
    .str.replace('.', '', regex=False)
    .str.replace('STREET', 'ST', regex=False)
    .str.replace('ROAD', 'RD', regex=False)
    .str.replace('AVENUE', 'AVE', regex=False)
)

# Create address
df['Address'] = (
    df['Property unit number'].fillna('').astype(str) + ' ' +
    df['Property house number'].fillna('').astype(str) + ' ' +
    df['Property street name'].fillna('').astype(str) + ' ' +
    df['Property locality'].fillna('').astype(str)
).str.strip().str.upper()

df = df.drop_duplicates()

# Cleans locality data
df['Property locality'] = df['Property locality'].astype(str)
df['Property locality'] = df['Property locality'].str.strip().str.upper()
df['Property locality'] = df['Property locality'].str.replace(r'\s+', ' ', regex=True)
df['Property locality'] = df['Property locality'].str.replace(r'[^\w\s]', '', regex=True)
df['Property locality'] = df['Property locality'].replace("NAN", pd.NA)
df['Property locality'] = df['Property locality'].apply(lambda x: x if len(str(x)) > 2 else pd.NA)
df['Property locality'] = df['Property locality'].str.replace(r'\d+', '', regex=True)


# Extracts time data from contract date
df['Year'] = df['Contract date'].dt.year
df['Month'] = df['Contract date'].dt.to_period('M')
df['Quarter'] = df['Contract date'].dt.to_period('Q')

# Organise primary purpose of properties
df = df.dropna(subset=['Primary purpose'])
df = df[df['Primary purpose'].isin(df['Primary purpose'].value_counts()[lambda x: x >= 20000].index)]

# Remove purchase price outliers
group_medians = (
    df.groupby(['Property post code', 'Primary purpose'])['Purchase price'].median()
)
df['group_median_price'] = df.set_index(['Property post code', 'Primary purpose']).index.map(group_medians)
df = df[df['Purchase price'] < df['group_median_price'] * 10]
df = df[df['Purchase price'] > df['group_median_price'] * 0.1]
df = df.drop(columns=['group_median_price'])

df.to_parquet('data/nsw_property_cleaned.parquet')
