"""Fit site parameters and their relationship with sand fraction.
Input: selected NetCDFs. Output: parameter CSVs in outputs/fit."""
from config import STATION_FILE, OUTPUT_ROOT, FIT_INPUT_DIR
import pandas as pd
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

OUTPUT_DIR = OUTPUT_ROOT / "fit"


def find_common_periods(list1, list2):
    common_periods = []
    start_end1 = list1.split('-')
    start_end2 = list2.split('-')
    latest_start = max(start_end1[0], start_end2[0])
    earliest_end = min(start_end1[1], start_end2[1])
    if latest_start <= earliest_end:
        common_periods.append(f'{latest_start}-{earliest_end}')
    else:
        common_periods.append(None)
    return common_periods


def model_s92(x, b):
    """Predict beta from relative wetness x[:, 0] and resistance r_d (s/m), x[:, 1]."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    # S92 soil resistance is converted to beta through the series resistance relation.
    y1 = np.exp(8.206 - b * x1)
    y2 = 1 / (1 + y1 / x2)
    return y2


def linear_func(x, a, b):
    return a * x + b


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stnlist = str(STATION_FILE)
    station_lists = pd.read_excel(stnlist, header=0)
    station_lists = station_lists[station_lists['run_flag'] == 1]
    # Indices in the run_flag==1 subset: US-AR1, US-SRG, US-Ton, US-Wkg.
    site = [2, 3, 5, 7]
    parameter_rows = []
    for num in range(4):
        i = site[num]
        sta_list = station_lists.iloc[i]
        sitename = sta_list['sitename']
        startend = sta_list['start_end']
        path_nc_SE = sta_list['start_end_qc']
        common_parts = find_common_periods(path_nc_SE, startend)
        path_nc_obs = str(FIT_INPUT_DIR) + "/" + sitename + '_' + common_parts[0] + '_E_ET_20250115.nc'
        data_obs = xr.open_dataset(path_nc_obs, engine='netcdf4')
        df_obs = data_obs.to_dataframe()
        mask1 = (df_obs['LE_CORR_E'] > 0) & (df_obs['Wsurf'] / df_obs['porsl'] < 0.6)
        df_obs.loc[~mask1] = np.nan
        df_obs.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df_obs.dropna()
        x_data = df_clean['Wsurf'] / df_clean['porsl']
        y_data = df_clean['beita']
        # Bin wetness at 0.05 intervals before selecting the fitting range.
        start, end, step = (0.0, 0.6, 0.05)
        bins = np.arange(start, end + step, step)
        labels = [f'{round(i, 3)}-{round(i + step, 3)}' for i in bins[:-1]]
        binned_data = pd.cut(x_data, bins=bins, labels=labels, include_lowest=True)
        df = pd.DataFrame({'x': x_data, 'y': y_data, 'bin': binned_data})
        df_clean['bin'] = df['bin']
        df_clean['x'] = df['x']
        # Exclude bins with fewer than 50 samples.
        min_samples = 50
        filtered_df = df_clean.groupby('bin').filter(lambda x: len(x) >= min_samples)
        x1 = filtered_df['Wsurf'] / filtered_df['porsl']
        # Fit b to observed beta for 0.35 <= wetness < 0.6 (upper bound filtered above).
        mask = (x1 >= 0.35) & (x1 <= 0.6)
        x2 = filtered_df['rd'][mask]
        x1 = x1[mask]
        ydata = filtered_df['beita'][mask]
        xdata = np.column_stack((x1, x2))
        popt, _ = curve_fit(model_s92, xdata, ydata)
        b_estimated = popt
        parameter_rows.append({'sitename': sitename, 'sand': float(filtered_df['sand'].iloc[0]), 'b_estimated': float(b_estimated[0]), 'n_fit': int(len(ydata))})
    full_df = pd.DataFrame(parameter_rows)
    full_df.to_csv(OUTPUT_DIR / 'S92_sand_station_parameters.csv', index=False)
    # Regress the fitted site parameters against sand fraction: b = slope*sand + intercept.
    x_data = full_df['sand'].values.astype(float)
    y_data = full_df['b_estimated'].values.astype(float)
    popt, _ = curve_fit(linear_func, x_data, y_data)
    r_squared = pearsonr(x_data, y_data)[0] ** 2
    pd.DataFrame([{'slope': popt[0], 'intercept': popt[1], 'R_squared': r_squared}]).to_csv(OUTPUT_DIR / 'S92_sand_coefficients.csv', index=False)
    print('S92_sand: b =', popt[0], '* sand +', popt[1], '; R2 =', r_squared)


if __name__ == "__main__":
    main()
