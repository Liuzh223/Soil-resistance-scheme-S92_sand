"""Select evaporation samples from outputs/derived.
Output: selected NetCDFs for parameter fitting and pooled X/Y arrays.
X stores the physical variables, including beta; Y stores the target beta."""
from config import STATION_FILE, OUTPUT_ROOT, DERIVED_INPUT_DIR
import pandas as pd
import numpy as np
import xarray as xr

OUTPUT_DIR = OUTPUT_ROOT / "samples"
SELECTED_OUTPUT_DIR = OUTPUT_ROOT / "selected"


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stnlist = str(STATION_FILE)
    station_lists = pd.read_excel(stnlist, header=0)
    station_lists = station_lists[station_lists['run_flag'] == 1]
    xx = len(station_lists)
    for i in range(xx):
        sta_list = station_lists.iloc[i]
        sitename = sta_list['sitename']
        startend = sta_list['start_end']
        path_nc_SE = sta_list['start_end_qc']
        common_parts = find_common_periods(path_nc_SE, startend)
        path_name = str(DERIVED_INPUT_DIR) + "/" + sitename + '_' + common_parts[0] + '_20250115.nc'
        data = xr.open_dataset(path_name, engine='netcdf4')
        df = data.to_dataframe()
        # Compute weak-signal thresholds from complete records.
        df_new_2 = df.dropna()
        if len(df_new_2) > 0:
            diff_q_max = np.percentile(df_new_2['diff_q'], 10)
            LE_CORR_max = np.percentile(df_new_2['LE_CORR'], 10)
        else:
            diff_q_max = LE_CORR_max = 0
        # Select E/ET >= 0.9, QC=0, wind >= 2 m/s, snow-free conditions, and wetness <= 0.6.
        mask1 = (df['ratio'] >= 0.9) & (df['SWC_F_MDS_1_QC'] == 0) & (df['TS_F_MDS_1_QC'] == 0) & ((df['diff_q'] >= np.minimum(diff_q_max, 0.05)) | (df['LE_CORR'] >= np.minimum(LE_CORR_max, 10))) & ((df['Wsurf'] >= df_new_2['wp'].iloc[0]) | (df['rss'] >= 100)) & ((df['Wsurf'] <= df_new_2['fc'].iloc[0]) | (df['rss'] <= 100)) & (df['Wind'] >= 2) & (df['mfsno'] <= 0) & (df['Wsurf'] / df_new_2['porsl'].iloc[0] > 0.0) & (df['Wsurf'] / df_new_2['porsl'].iloc[0] <= 0.6)
        df.loc[~mask1] = np.nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df.dropna()
        # Save selected records for the subsequent parameter fitting.
        xr.Dataset.from_dataframe(df).to_netcdf(SELECTED_OUTPUT_DIR / (sitename + "_" + common_parts[0] + "_E_ET_20250115.nc"))
        print('------------')
        print(sitename)
        x_data = df_clean['Wsurf'] / df_clean['porsl']
        y_data = df_clean['beita']
        # Bin relative soil wetness at 0.05 intervals, separately for each site.
        start, end, step = (0.0, 0.6, 0.05)
        bins = np.arange(start, end + step, step)
        labels = [f'{round(i, 3)}-{round(i + step, 3)}' for i in bins[:-1]]
        binned_data = pd.cut(x_data, bins=bins, labels=labels, include_lowest=True)
        df = pd.DataFrame({'x': x_data, 'y': y_data, 'bin': binned_data})
        df_clean['bin'] = df['bin']
        df_clean['x'] = df['x']
        print('Sample count in each bin:')
        print(df_clean['bin'].value_counts())
        # Remove bins containing fewer than 50 samples.
        min_samples = 50
        filtered_df = df_clean.groupby('bin').filter(lambda x: len(x) >= min_samples)
        print('Sample count in each bin after filtering:')
        print(filtered_df['bin'].value_counts())
        df_clean = filtered_df
        swc = df_clean['Wsurf']
        porsl = df_clean['porsl']
        aird = df_clean['aird']
        fc = df_clean['fc']
        um = df_clean['Wind']
        swt = df_clean['Tsurf']
        rd = df_clean['rd']
        dg = df_clean['dg']
        Qg = df_clean['Qg']
        beta = df_clean['beita']
        bsw = df_clean['bsw']
        psi0 = df_clean['psi0']
        hksati = df_clean['hksati']
        PET = df_clean['PET']
        sand = df_clean['sand']
        om = df_clean['om']
        porsl = porsl.astype(np.float32)
        aird = aird.astype(np.float32)
        fc = fc.astype(np.float32)
        sand = sand.astype(np.float32)
        om = om.astype(np.float32)
        um = um.astype(np.float32)
        swt = swt.astype(np.float32)
        dg = dg.astype(np.float32)
        rd = rd.astype(np.float32)
        PET = PET.astype(np.float32)
        beta = beta.astype(np.float32)
        Qg = Qg.astype(np.float32)
        bsw = bsw.astype(np.float32)
        psi0 = psi0.astype(np.float32)
        hksati = hksati.astype(np.float32)
        # X rows: swc,porsl,aird,fc,sand,om,um,swt,dg,rd,PET,beta,Qg,bsw,psi0,hksati.
        X = np.stack((swc, porsl, aird, fc, sand, om, um, swt, dg, rd, PET, beta, Qg, bsw, psi0, hksati))
        Y = beta
        # Concatenate site samples in the listed variable order.
        if i == 0:
            x_train = X
            y_train = Y
        else:
            x_train = np.concatenate((x_train, X), axis=1)
            y_train = np.concatenate((y_train, Y), axis=0)
        print(f'------{sitename}-------------')
        print(len(Y))
    np.save(str(OUTPUT_DIR / "X_hourly_20250115_beta_all_corr.npy"), x_train)
    np.save(str(OUTPUT_DIR / "Y_hourly_20250115_beta_all_corr.npy"), y_train)


if __name__ == "__main__":
    main()
