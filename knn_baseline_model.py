from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd
import sklearn
import tqdm
from tqdm.auto import tqdm
import scipy.stats
import sklearn.neighbors

df = pd.read_pickle('finish_satellite.pkl')
df['survey'] = df['DHSID_EA'].str[:10]
df['cc'] = df['DHSID_EA'].str[:2]
path_years = df[['DHSID_EA', 'path', 'year']].apply(tuple, axis=1)
print(df['path'].iloc[0])
print(df.head())

label_cols = ['water_index']

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

# np.load(npz_path) is object
# ['x'] get all the values: shape 8, 255, 255
# -1: just last channel 
# last band are nightbands from DMSP or VIIRS

results_df = pd.DataFrame(
    data=np.nan,
    columns=['nl_mean', 'year'],
    index=pd.Index(sorted(df['DHSID_EA']), name='DHSID_EA')
)
print(results_df.head())

df.set_index('DHSID_EA', verify_integrity=True, inplace=True)

with ThreadPoolExecutor(max_workers=30) as pool:
    inputs = path_years
    futures = pool.map(calculate_nl_mean, inputs)
    for dhsid_ea, nl_mean, year in tqdm(futures, total=len(inputs)):
        results_df.loc[dhsid_ea, ['nl_mean', 'year']] = (nl_mean, year)

print(results_df.head())

results_df.to_csv('mean_nl.csv')
results_df['year'] = results_df['year'].astype(int)

SPLITS = {
    'train': [
        'AL', 'BD', 'CD', 'CM', 'GH', 'GU', 'HN', 'IA', 'ID', 'JO', 'KE', 'KM',
        'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 'NG', 'NI', 'PE', 'PH',
        'SN', 'TG', 'TJ', 'UG', 'ZM', 'ZW'],
    'val': [
        'BF', 'BJ', 'BO', 'CO', 'DR', 'GA', 'GN', 'GY', 'HT', 'NM', 'SL', 'TD',
        'TZ'],
    'test': [
        'AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'KY', 'ML', 'NP', 'PK', 'RW',
        'SZ']
}
SPLITS['trainval'] = SPLITS['train'] + SPLITS['val']

results_df['cc'] = results_df.index.str[:2]

# trennung zwischen DMSP und VIIRS 
def run(knn, label, dmsp, trainsplit='train', testsplit='test'):
    if dmsp:
        year_mask = (df['year'] <= 2011)
    else:
        year_mask = (df['year'] > 2011)

    train_dhsids = df.index[year_mask & df['cc'].isin(SPLITS[trainsplit]) & df[label].notna()]
    test_dhsids = df.index[year_mask & df['cc'].isin(SPLITS[testsplit]) & df[label].notna()]

    train_X = results_df.loc[train_dhsids, 'nl_mean'].values.reshape(-1, 1)
    train_Y = df.loc[train_dhsids, label].values
    test_X = results_df.loc[test_dhsids, 'nl_mean'].values.reshape(-1, 1)
    test_Y = df.loc[test_dhsids, label].values

    knn.fit(train_X, train_Y)
    preds = knn.predict(test_X)
    return preds, test_Y


for label in label_cols:
    print(f'=========== {label:15s} ============')
    best_r2 = 0
    best_k = None
    for k in range(1, 21):
        knn = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=k, weights='uniform', algorithm='auto')
        preds_dmsp, labels_dmsp = run(knn, label, True, 'train', 'val')
        preds_viirs, labels_viirs = run(knn, label, False, 'train', 'val')
        r2 = scipy.stats.pearsonr(
            np.concatenate([preds_dmsp, preds_viirs]),
            np.concatenate([labels_dmsp, labels_viirs])
        )[0]**2
        print(f'k={k:2d}, {label:15s} r^2 = {r2:.3f}')
        if r2 > best_r2:
            best_r2 = r2
            best_k = k
    knn = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=best_k, weights='uniform', algorithm='auto')
    preds_dmsp, labels_dmsp = run(knn, label, True, 'trainval', 'test')
    preds_viirs, labels_viirs = run(knn, label, False, 'trainval', 'test')
    r2 = scipy.stats.pearsonr(
        np.concatenate([preds_dmsp, preds_viirs]),
        np.concatenate([labels_dmsp, labels_viirs])
    )[0]**2
    print(f'FINAL: k={best_k:2d}, {label:15s} r^2 = {r2:.2f}')    

    # k=6, r^2 = 0.404