from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd
import sklearn
#from tqdm.auto import tqdm

df = pd.read_pickle('finish_satellite.pkl')
df['survey'] = df['DHSID_EA'].str[:10]
df['cc'] = df['DHSID_EA'].str[:2]
path_years = df[['DHSID_EA', 'path', 'year']].apply(tuple, axis=1)

# hier DHSID_EA als index und damit nicht mehr unter keys 
df.set_index('DHSID_EA', verify_integrity=True, inplace=True)
print(df['path'].iloc[0])
print(df.head())

label_cols = ['asset_index', 'under5_mort', 'women_bmi', 'women_edu', 'water_index', 'sanitation_index']

def calculate_nl_mean(path_and_year) -> tuple[np.ndarray, np.ndarray, int]:
    '''
    Args
    - path_year: tuple (path, year)
      - path: str, path to npz file containing single entry 'x'
        representing a (C, H, W) image
      - year: int

    Returns: (nl_mean, year)
    '''
    dhsid_ea, npz_path, year = path_and_year
    img = np.load(npz_path)['x']  # shape (C, H, W)
    nl_mean = img[-1].mean(dtype=np.float64)
    return dhsid_ea, nl_mean, year

print(df.keys())

results_df = pd.DataFrame(
    data=np.nan,
    columns=['nl_mean', 'year'],
    index=pd.Index(sorted(df['DHSID_EA']), name='DHSID_EA')
)
print(results_df.head())


with ThreadPoolExecutor(max_workers=30) as pool:
    inputs = path_years
    futures = pool.map(calculate_nl_mean, inputs)
    for dhsid_ea, nl_mean, year in tqdm(futures, total=len(inputs)):
        results_df.loc[dhsid_ea, ['nl_mean', 'year']] = (nl_mean, year)

