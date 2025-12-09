import pandas as pd
from pathlib import Path

# Filters to residential property only
# Filters columns to only keep neccessary features
# Split data into training, validation and testing
def build_model_dataset():
    print("Loading data...")
    df = pd.read_parquet('data/nsw_property_features.parquet')

    df = df[df['Primary purpose'] == 'Residence'].copy()

    featured_columns = [
        'Purchase price',
        'Area', 
        'price_per_m',
        'suburb_median_price',
        'suburb_price_growth', 
        'zoning_median_price',
        'zoning_growth',
        'sale_year', 
        'sale_month', 
        'sale_quarter',
        'Year', 
        'Property post code',
        'Zoning',
        'price_log',
    ]

    featured_columns = [c for c in featured_columns if c in df.columns]
    df = df[featured_columns].copy()

    df = df.sort_values('Year')

    print("Saving data...")
    df.to_parquet('data/model_data.parquet')


if __name__ == "__main__":
    build_model_dataset()
