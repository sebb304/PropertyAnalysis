import pandas as pd
import numpy as np

def create_time_feature(df):
    df['Contract date'] = pd.to_datetime(df['Contract date'], errors='coerce')
    df['sale_year'] = df['Contract date'].dt.year
    df['sale_month'] = df['Contract date'].dt.month
    df['sale_quarter'] = df['Contract date'].dt.quarter
    df['sale_year_month'] = df['Contract date'].dt.to_period('M').astype(str)

    return df

def create_price_feature(df):
    # Log transform of price
    df['price_log'] = np.log1p(df['Purchase price'])

    df['price_per_m'] = df['Purchase price'] / df['Area'].replace({0: np.nan})

    df = df.sort_values(['Property locality', 'Contract date'])
    df['suburb_median_price'] = (
        df.groupby('Property locality')['Purchase price']
        .transform(lambda x: x.rolling(window=12, min_periods=1).median())
    )

    df['suburb_yearly_median'] = (
        df.groupby(['Property locality', 'sale_year'])['Purchase price']
        .transform('median')
    )

    df['suburb_price_growth'] = (
        df.groupby('Property locality')['suburb_yearly_median']
        .pct_change()
    )

    return df

def create_suburb_activity_feature(df):
    df['suburb_sales_per_year'] = (
        df.groupby(['Property locality', 'sale_year'])['Purchase price']
        .transform('count')
    )

    return df

def create_zoning_feature(df):
    df['zoning_median_price'] = (
        df.groupby('Zoning')['Purchase price']
        .transform('median')
    )
    df['zoning_growth'] = (
        df.groupby('Zoning')['zoning_median_price']
        .pct_change()
    )

    return df

def create_repeated_sales_feature(df):
    df = df.sort_values(['Address', 'Contract date'])

    df['prev_sale_price'] = df.groupby('Address')['Purchase price'].shift(1)
    df['price_change'] = df['Purchase price'] - df['prev_sale_price']
    df['price_pct_change'] = df['price_change'] / df['prev_sale_price']

    df['prev_sale_date'] = df.groupby('Address')['Contract date'].shift(1)
    df['years_between_sale'] = (df['Contract date'] - df['prev_sale_date']).dt.days / 365

    return df

def create_outlier_feature(df):
    df['suburb_median'] = df.groupby('Property locality')['Purchase price'].transform('median')
    df['price_high_outlier'] = df['Purchase price'] > df['suburb_median'] * 4
    df['price_low_outlier'] = df['Purchase price'] < df['suburb_median'] * 0.25

    df['landsize_high_outlier'] = df.groupby('Property locality')['Area'].transform(lambda x: x > x.quantile(0.99)).fillna(False)
    df['landsize_low_outlier'] = df.groupby('Property locality')['Area'].transform(lambda x: x < x.quantile(0.01)).fillna(False)

    return df

def main():
    print("Loading cleaned data...")
    df = pd.read_parquet('data/nsw_property_cleaned.parquet')

    print("Creating time features...")
    df = create_time_feature(df)

    print("Creating price features...")
    df = create_price_feature(df)

    print("Creating suburb activity features...")
    df = create_suburb_activity_feature(df) 

    print("Creating zoning features...")
    df = create_zoning_feature(df) 

    print("Creating repeated sales features...")
    df = create_repeated_sales_feature(df)

    print("Saving data...")
    df.to_parquet('data/nsw_property_features.parquet')


if __name__ == "__main__":
    main()