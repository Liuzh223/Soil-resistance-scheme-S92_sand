"""Derive surface conditions and beta/r_ss using CoLM simulations with the S92 scheme."""
from config import DATA_ROOT, STATION_FILE, OUTPUT_ROOT, BASE_INPUT_DIR, BASE_FILE_PATTERN
import pandas as pd
import numpy as np
import xarray as xr
import physical_utils as atmos

OUTPUT_DIR = OUTPUT_ROOT / "derived"


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
        path_nc_file = BASE_INPUT_DIR / BASE_FILE_PATTERN.format(sitename=sitename, period=common_parts[0])
        data_base = xr.open_dataset(path_nc_file, engine='netcdf4')
        df = data_base.to_dataframe()
        path_ATR = (DATA_ROOT / "Fluxnet2015").as_posix() + "/" + 'FLX_' + sitename + '_FLUXNET2015_FULLSET_DD_' + startend + '.csv'
        data_file = pd.read_csv(path_ATR)
        data_file.set_index('TIMESTAMP', inplace=True)
        data_file.index = pd.to_datetime(data_file.index, format='%Y%m%d')
        common_parts = find_common_periods(path_nc_SE, startend)
        start_year, end_year = [part.split('-') for part in common_parts][0]
        start_date = f'{start_year}-01-01 00:00'
        end_date = f'{end_year}-12-31 23:30'
        print('LW_OUT_QC' in data_file)
        if 'LW_OUT_QC' in data_file:
            data_f = data_file[start_date:end_date]
            date_range = pd.date_range(start=start_date, end=end_date, freq='30T')
            LW_OUT_QC = data_f['LW_OUT_QC'].reindex(date_range).ffill()
            df['LW_OUT_QC'] = LW_OUT_QC[0:len(df)]
            df.loc[df['LW_OUT_QC'] != 1, 'LW_OUT_QC'] = np.nan
            del data_f
        del data_file
        Psurf = df['Psurf']
        Qair = df['Qair']
        Tair = df['Tair']
        LWdown = df['LWdown']
        LE_CORR = df['LE_CORR']
        Ustar = df['Ustar']
        mUstar = df['mUstar']
        LAI = df['LAI_N']
        SAI = df['SAI_N']
        TS = df['TS']
        SWC = df['SWC']
        Ts_model_1 = df['Ts_model_1']
        Ts_model_2 = df['Ts_model_2']
        Ts_model_3 = df['Ts_model_3']
        SWC_model_1 = df['SWC_model_1']
        SWC_model_2 = df['SWC_model_2']
        SWC_model_3 = df['SWC_model_3']
        SWC_model_4 = df['SWC_model_4']
        evpa = df['evpa']
        evpg = df['evpg']
        Tleaf = df['Tleaf']
        mLwup = df['mLwup']
        TS_height = df['TS_height'][0]
        SWC_height = df['SWC_height'][0]
        IGBP = df['IGBP'][0]
        # Infer surface temperature from upward longwave radiation where available.
        if 'LW_OUT' in df:
            LWout = df['LW_OUT']
            df['Tsurf'] = atmos.calc_surface_temperature_from_canopy_obs_ratio(LWdown, mLwup, LWout, Tleaf, 0.96, IGBP, LAI, SAI)
            if 'LWup' in df:
                LWup = df['LWup']
                df['Tsurf'] = atmos.calc_surface_temperature_from_canopy_obs_ratio(LWdown, mLwup, LWup, Tleaf, 0.96, IGBP, LAI, SAI)
        else:
            if 0 <= abs(TS_height) <= 0.02:
                df['Tsurf'] = TS
            if 0.02 < abs(TS_height) <= 0.05:
                df['Tsurf'] = TS * (Ts_model_1 / Ts_model_2)
            if 0.05 < abs(TS_height) <= 0.1:
                df['Tsurf'] = TS * (Ts_model_1 / Ts_model_3)
        # Adjust moisture using the model ratio between surface and measurement-depth layers.
        if 0 <= abs(SWC_height) <= 0.02:
            df['Wsurf'] = SWC
        if 0.02 < abs(SWC_height) <= 0.05:
            df['Wsurf'] = SWC * (SWC_model_1 / SWC_model_2)
        if 0.05 < abs(SWC_height) <= 0.1:
            df['Wsurf'] = SWC * (SWC_model_1 / SWC_model_3)
        if 0.1 < abs(SWC_height) <= 0.5:
            df['Wsurf'] = SWC * (SWC_model_1 / SWC_model_4)
        # For measurements within 2 cm of the surface, use observed TS directly.
        if 0 <= abs(TS_height) <= 0.02:
            df['Tsurf'] = TS
        df['Ustar'] = np.maximum(1e-06, df['Ustar'])
        # Soil-pore specific humidity (kg/kg), accounting for soil water potential.
        df['Qg'] = df.apply(lambda x: atmos.calculate_Qg(x['Qair'], x['Tsurf'], x['Psurf'], x['psi0'], x['Wsurf'], x['porsl'], x['bsw']), axis=1)
        # Air density, moisture thresholds, and effective vapor diffusivity.
        df['rhoair'] = atmos.calculate_rhoair(Psurf, Qair, Tair)
        df['aird'] = atmos.soilress_aird(df['porsl'], df['bsw'], df['psi0'])
        df['wp'] = atmos.soilress_wp(df['porsl'], df['bsw'], df['psi0'])
        df['fc'] = atmos.soilress_fc(df['porsl'], df['bsw'], df['psi0'])
        df['dg'] = atmos.soilress_dg(df['porsl'], df['bsw'], df['psi0'], df['Tsurf'])
        df['Lambda'] = 2510400.0
        # Aerodynamic and surface-to-canopy-air resistances (s/m).
        df['raw'] = df.apply(lambda row: atmos.calculate_raw(row['Tair'], row['Psurf'], row['LAI_N'], row['SAI_N'], row['Qair'], row['rhoair'], row['Ustar'], row['LE_CORR'], row['H_CORR'], row['ht'], row['Hcan']), axis=1)
        df['rd'] = atmos.calc_rd(Ustar, LAI, SAI)
        df['mrd'] = atmos.calc_rd(mUstar, LAI, SAI)
        # Infer canopy-air specific humidity from the observed latent heat flux.
        df['Qaf'] = LE_CORR * df['raw'] / (df['Lambda'] * df['rhoair']) + df['Qair']
        ratio = evpg / evpa
        ratio = np.minimum(1, ratio)
        ratio = np.maximum(0, ratio)
        df['ratio'] = ratio[0:len(df)]
        # Partition LE using E/ET, then derive soil resistance and beta.
        df['r_all'] = df['rhoair'] * df['Lambda'] * (df['Qg'] - df['Qaf']) / (LE_CORR * ratio)
        df['rss'] = df['r_all'] - df['rd']
        df['diff_q'] = df['Qg'] - df['Qaf']
        df['dsl'] = df['rss'] * df['dg']
        df['beita'] = 1 / (1 + df['rss'] / df['rd'])
        df['mbeita'] = 1 / (1 + df['mrss'] / df['mrd'])
        df['Wsurf_R'] = df['Wsurf'] / df['porsl']
        df['LE_CORR_E'] = df['LE_CORR'] * ratio
        df['PET'] = df['rhoair'] * df['Lambda'] * (df['Qg'] - df['Qaf']) / df['rd']
        # Retain this diagnostic: its missing values affect sample filtering.
        df['mLE_E'] = df['mLE'] * (df['evpa'] / df['evpg'])
        df['mPET'] = df['mLE_E'] / df['mbeita']
        df_new_1 = df.dropna()
        # Apply physical/QC limits; keep timestamps by setting rejected rows to NaN.
        mask1 = (df['beita'] > 0) & (df['year_qc'] == 0) & (df['LAI_N'] <= 1.5) & (df['r_all'] > 0) & (df['rss'] >= 0) & (df['rss'] <= 10000.0) & (df['Wsurf'] >= df_new_1['aird'].iloc[0]) & (df['Wsurf'] <= df_new_1['porsl'].iloc[0]) & (df['dsl'] > 0) & (df['dsl'] <= 0.05)
        df.loc[~mask1] = np.nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df.dropna()
        print('------------')
        print(sitename)
        print(len(df))
        if len(df_clean) > 2:
            ds = xr.Dataset.from_dataframe(df)
            nc_filename = str(OUTPUT_DIR) + "/" + sitename + '_' + common_parts[0] + '_20250115.nc'
            ds.to_netcdf(path=nc_filename, mode='w')
            print(f'Data saved to {nc_filename}')


if __name__ == "__main__":
    main()
