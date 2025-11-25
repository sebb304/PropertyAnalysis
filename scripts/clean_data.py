import pandas as pd

df = pd.read_csv('/Users/seb/Desktop/Project2/nsw_property_data.csv', low_memory=False)


# Convert to date type
df['Contract date'] = pd.to_datetime(df['Contract date'], errors='coerce')
df['Settlement date'] = pd.to_datetime(df['Settlement date'], errors='coerce')

# Convert to int
df['Property post dode'] = df['Property post code'].astype('Int64')
df['Purchase price'] = df['Purchase price'].astype('Int64')

df = df[df['Purchase price'].notna()]
df = df[df['Purchase price'] > 0]

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

df['LGA_ID'] = (
    df['Property legal description'].fillna('') + '|' +
    df["Address"]
)

df = df.drop_duplicates(
    subset=['LGA_ID', 'Contract date', 'Purchase price'],
    keep='first'
)

# Extracts time data from contract date
df['Year'] = df['Contract date'].dt.year
df['Month'] = df['Contract date'].dt.to_period('M')
df['Quarter'] = df['Contract date'].dt.to_period('Q')

# Removes entries with location missing
df[df['Property locality'].isna() | df['Property post code'].isna()]

df.to_parquet('data/nsw_property_cleaned.parquet')
