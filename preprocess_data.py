"""Prepare observations and CoLM S92 auxiliary data.
Output: outputs/base_data, read by derive_soil_evaporation.py."""
from config import DATA_ROOT, STATION_FILE, OUTPUT_ROOT, MODEL_INPUT_DIR, MODEL_FILE_PATTERN, BASE_FILE_PATTERN
import pandas as pd
import numpy as np
import xarray as xr

OUTPUT_DIR = OUTPUT_ROOT / "base_data"


def get_height(sitename):
    """Read TS/SWC measurement depths (m) from the site metadata."""
    df = pd.read_excel(str(DATA_ROOT / "csv_file/Fluxnet-BADM.xlsx"))
    site_id_column = 'SITE_ID'
    data_value_column = 'DATAVALUE'
    matching_rows = df[df[site_id_column] == sitename]
    new_index_ts = matching_rows[matching_rows[data_value_column] == 'TS_F_MDS_1'].index
    if not new_index_ts.empty:
        TS_height = df[data_value_column][new_index_ts[0] + 1]
        TS_height = float(TS_height) if pd.notnull(TS_height) else np.nan
    else:
        TS_height = np.nan
    new_index_swc = matching_rows[matching_rows[data_value_column] == 'SWC_F_MDS_1'].index
    if not new_index_swc.empty:
        SWC_height = df[data_value_column][new_index_swc[0] + 1]
        SWC_height = float(SWC_height) if pd.notnull(SWC_height) else np.nan
    else:
        SWC_height = np.nan
    # Use the available depth if only one measurement depth is missing.
    if np.isnan(TS_height) and (not np.isnan(SWC_height)):
        TS_height = SWC_height
    elif np.isnan(SWC_height) and (not np.isnan(TS_height)):
        SWC_height = TS_height
    elif np.isnan(TS_height) and np.isnan(SWC_height):
        TS_height = np.nan
        SWC_height = np.nan
    return (TS_height, SWC_height)


def clean_column(column, qc_column=None, qc_standard=1, invalid_value=-9999):
    """Mask values that fail QC or equal the missing-value sentinel."""
    if qc_column is not None:
        column[(qc_column > qc_standard) | (column == invalid_value)] = np.nan
    else:
        column[column == invalid_value] = np.nan
    return column


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
    Lamda = 2510400.0
    xx = len(station_lists)
    for i in range(xx):
        sta_list = station_lists.iloc[i]
        sitename = sta_list['sitename']
        startend = sta_list['start_end']
        path_nc_SE = sta_list['start_end_qc']
        source = sta_list['source']
        # Read FLUXNET observations, meteorological forcing, and site properties.
        path_ATR = (DATA_ROOT / "Fluxnet2015").as_posix() + "/" + 'FLX_' + sitename + '_FLUXNET2015_FULLSET_HH_' + startend + '.csv'
        path_FLX = (DATA_ROOT / "Flux").as_posix() + "/" + sitename + '_' + path_nc_SE + '_' + source + '_Flux.nc'
        path_MET = (DATA_ROOT / "Forcing").as_posix() + "/" + sitename + '_' + path_nc_SE + '_' + source + '_Met.nc'
        path_SRF = (DATA_ROOT / "Sitedata").as_posix() + "/" + sitename + '_' + path_nc_SE + '_' + source + '_Site.nc'
        path_model = MODEL_INPUT_DIR / MODEL_FILE_PATTERN.format(sitename=sitename, start_year=path_nc_SE.split('-')[0], end_year=path_nc_SE.split('-')[1])
        data_file = pd.read_csv(path_ATR)
        data_file.set_index('TIMESTAMP_START', inplace=True)
        data_file.index = pd.to_datetime(data_file.index, format='%Y%m%d%H%M')
        data_flx = xr.open_dataset(path_FLX, engine='netcdf4')
        data_flx['time'] = pd.to_datetime(data_flx['time'].values, format='%Y%m%d%H%M')
        if 'LWup' in data_flx:
            data_flx = data_flx[['Qle_cor', 'Qh_cor', 'Ustar', 'LWup', 'Qle', 'Qh']]
        else:
            data_flx = data_flx[['Qle_cor', 'Qh_cor', 'Ustar', 'Qle', 'Qh']]
        data_met = xr.open_dataset(path_MET, engine='netcdf4')
        data_met['time'] = pd.to_datetime(data_met['time'].values, format='%Y%m%d%H%M')
        data_met = data_met[['Psurf', 'Qair', 'Tair', 'Wind', 'LWdown', 'Precip', 'reference_height_q']]
        data_srf = xr.open_dataset(path_SRF, engine='netcdf4')
        data_model = xr.open_dataset(path_model, engine='netcdf4')
        data_model['time'] = pd.to_datetime(data_model['time'].values, format='%Y%m%d%H%M')
        data_model = data_model[['f_t_soisno', 'f_h2osoi', 'f_fevpa', 'f_fevpg', 'f_tleaf', 'f_olrg', 'f_t_grnd', 'f_ustar', 'f_lfevpa', 'f_fsena', 'f_fsno', 'f_lai', 'f_sai', 'f_rss']]
        # Restrict observations and model outputs to their common years.
        common_parts = find_common_periods(path_nc_SE, startend)
        start_year, end_year = [part.split('-') for part in common_parts][0]
        print(sitename)
        print(start_year, end_year)
        start_date = f'{start_year}-01-01 00:00'
        end_date = f'{end_year}-12-31 23:30'
        data_swc = data_file[start_date:end_date]
        data_flx = data_flx.where((data_flx['time'].dt.year >= int(f'{start_year}')) & (data_flx['time'].dt.year <= int(f'{end_year}')), drop=True)
        data_met = data_met.where((data_met['time'].dt.year >= int(f'{start_year}')) & (data_met['time'].dt.year <= int(f'{end_year}')), drop=True)
        data_model = data_model.where((data_model['time'].dt.year >= int(f'{start_year}')) & (data_model['time'].dt.year <= int(f'{end_year}')), drop=True)
        data_swc['year'] = data_swc.index.year
        sele_years = data_srf['year'][data_srf['year_qc'] == 1]
        data_swc['year_qc'] = data_swc['year'].map(lambda x: 0 if x in sele_years else np.nan)
        # Soil moisture and sand/organic content are volumetric fractions.
        ht = data_met['reference_height_q'].values[0]
        porsl = data_srf['soil_theta_s'].values[0]
        # Convert soil water potential from cm to mm and conductivity to mm/s.
        psi0 = data_srf['soil_psi_s'].values[0] * 10.0
        bsw = 1.0 / data_srf['soil_lambda'].values[0]
        hksati = data_srf['soil_k_s'].values[0] * 10.0 / 86400.0
        om = data_srf['soil_vf_om'].values[0]
        sand = data_srf['soil_vf_sand'].values[0]
        Hcan = data_srf['canopy_height'].values
        IGBP = data_srf['IGBP_classification'].values
        lat = data_srf['latitude'].values
        lon = data_srf['longitude'].values
        TS_height, SWC_height = get_height(sitename)
        print(sitename, TS_height, SWC_height)
        Precip = data_met['Precip'][:, 0, 0].values
        Psurf = data_met['Psurf'][:, 0, 0].values
        Qair = data_met['Qair'][:, 0, 0].values
        Tair = data_met['Tair'][:, 0, 0].values
        Wind = data_met['Wind'][:, 0, 0].values
        LWdown = data_met['LWdown'][:, 0, 0].values
        LE_CORR = data_flx['Qle_cor'][:, 0, 0].values
        H_CORR = data_flx['Qh_cor'][:, 0, 0].values
        LE = data_flx['Qle'][:, 0, 0].values
        SH = data_flx['Qh'][:, 0, 0].values
        Ustar = data_flx['Ustar'][:, 0, 0].values
        # Extract model soil layers used to correct observed surface conditions.
        Ts_model_1 = data_model['f_t_soisno'][:, 0, 5].values
        Ts_model_2 = data_model['f_t_soisno'][:, 0, 6].values
        Ts_model_3 = data_model['f_t_soisno'][:, 0, 7].values
        # Prepend the first model value, then truncate to observation length.
        Ts_model_1 = np.insert(Ts_model_1, 0, Ts_model_1[0])
        Ts_model_2 = np.insert(Ts_model_2, 0, Ts_model_2[0])
        Ts_model_3 = np.insert(Ts_model_3, 0, Ts_model_3[0])
        SWC_model_1 = data_model['f_h2osoi'][:, 0, 0].values
        SWC_model_2 = data_model['f_h2osoi'][:, 0, 1].values
        SWC_model_3 = data_model['f_h2osoi'][:, 0, 2].values
        SWC_model_4 = data_model['f_h2osoi'][:, 0, 3].values
        SWC_model_1 = np.insert(SWC_model_1, 0, SWC_model_1[0])
        SWC_model_2 = np.insert(SWC_model_2, 0, SWC_model_2[0])
        SWC_model_3 = np.insert(SWC_model_3, 0, SWC_model_3[0])
        SWC_model_4 = np.insert(SWC_model_4, 0, SWC_model_4[0])
        # evpg is soil evaporation; evpa is total evapotranspiration.
        evpg = data_model['f_fevpg'][:, 0].values
        evpa = data_model['f_fevpa'][:, 0].values
        mLwup = data_model['f_olrg'][:, 0].values
        tleaf = data_model['f_tleaf'][:, 0].values
        mLE = data_model['f_lfevpa'][:, 0].values
        mH = data_model['f_fsena'][:, 0].values
        mUstar = data_model['f_ustar'][:, 0].values
        mfsno = data_model['f_fsno'][:, 0].values
        mrss = data_model['f_rss'][:, 0].values
        LAI = data_model['f_lai'][:, 0].values
        SAI = data_model['f_sai'][:, 0].values
        evpg = np.insert(evpg, 0, evpg[0])
        evpa = np.insert(evpa, 0, evpa[0])
        tleaf = np.insert(tleaf, 0, tleaf[0])
        mLwup = np.insert(mLwup, 0, mLwup[0])
        mLE = np.insert(mLE, 0, mLE[0])
        mH = np.insert(mH, 0, mH[0])
        mUstar = np.insert(mUstar, 0, mUstar[0])
        mfsno = np.insert(mfsno, 0, mfsno[0])
        mrss = np.insert(mrss, 0, mrss[0])
        LAI = np.insert(LAI, 0, LAI[0])
        SAI = np.insert(SAI, 0, SAI[0])
        if 'LWup' in data_flx:
            LWup = data_flx['LWup'][:, 0, 0].values
            data_swc['LWup'] = LWup[0:len(data_swc)]
        data_swc['Precip'] = Precip[0:len(data_swc)]
        data_swc['Tair'] = Tair[0:len(data_swc)]
        data_swc['Qair'] = Qair[0:len(data_swc)]
        data_swc['LE_CORR'] = LE_CORR[0:len(data_swc)]
        data_swc['H_CORR'] = H_CORR[0:len(data_swc)]
        data_swc['LE'] = LE[0:len(data_swc)]
        data_swc['SH'] = SH[0:len(data_swc)]
        data_swc['Psurf'] = Psurf[0:len(data_swc)]
        data_swc['Ustar'] = Ustar[0:len(data_swc)]
        data_swc['Wind'] = Wind[0:len(data_swc)]
        data_swc['LWdown'] = LWdown[0:len(data_swc)]
        data_swc['Ts_model_1'] = Ts_model_1[0:len(data_swc)]
        data_swc['Ts_model_2'] = Ts_model_2[0:len(data_swc)]
        data_swc['Ts_model_3'] = Ts_model_3[0:len(data_swc)]
        data_swc['SWC_model_1'] = SWC_model_1[0:len(data_swc)]
        data_swc['SWC_model_2'] = SWC_model_2[0:len(data_swc)]
        data_swc['SWC_model_3'] = SWC_model_3[0:len(data_swc)]
        data_swc['SWC_model_4'] = SWC_model_4[0:len(data_swc)]
        data_swc['evpg'] = evpg[0:len(data_swc)]
        data_swc['evpa'] = evpa[0:len(data_swc)]
        data_swc['Tleaf'] = tleaf[0:len(data_swc)]
        data_swc['mLwup'] = mLwup[0:len(data_swc)]
        data_swc['mLE'] = mLE[0:len(data_swc)]
        data_swc['mH'] = mH[0:len(data_swc)]
        data_swc['mUstar'] = mUstar[0:len(data_swc)]
        data_swc['mfsno'] = mfsno[0:len(data_swc)]
        data_swc['mrss'] = mrss[0:len(data_swc)]
        data_swc['LAI_N'] = LAI[0:len(data_swc)]
        data_swc['SAI_N'] = SAI[0:len(data_swc)]
        data_swc['SWC'] = clean_column(data_swc['SWC_F_MDS_1'], data_swc['SWC_F_MDS_1_QC'])
        data_swc['TS'] = clean_column(data_swc['TS_F_MDS_1'], data_swc['TS_F_MDS_1_QC'])
        # Convert observed temperature to K and soil moisture from percent to fraction.
        data_swc['TS'] = data_swc['TS'] + 273.15
        data_swc['SWC'] = data_swc['SWC'] / 100
        if 'LW_OUT' in data_swc:
            data_swc['LW_OUT'] = clean_column(data_swc['LW_OUT'])
        if 'LW_OUT' in data_swc:
            print('--------------')
            print(sitename)
            print('LW_out EXIST')
            if 'LWup' in data_flx:
                df_new = data_swc[['Precip', 'Psurf', 'Tair', 'Qair', 'Ustar', 'Wind', 'LE_CORR', 'SWC_F_MDS_1_QC', 'TS_F_MDS_1_QC', 'LAI_N', 'SAI_N', 'year_qc', 'TS', 'SWC', 'LWdown', 'LWup', 'LW_OUT', 'H_CORR', 'Ts_model_1', 'Ts_model_2', 'Ts_model_3', 'SWC_model_1', 'SWC_model_2', 'SWC_model_3', 'SWC_model_4', 'evpg', 'evpa', 'Tleaf', 'mLwup', 'mLE', 'mH', 'mUstar', 'mfsno', 'mrss', 'LE', 'SH']].copy()
                print('LWUP EXIST')
            else:
                df_new = data_swc[['Precip', 'Psurf', 'Tair', 'Qair', 'Ustar', 'Wind', 'LE_CORR', 'SWC_F_MDS_1_QC', 'TS_F_MDS_1_QC', 'LAI_N', 'SAI_N', 'year_qc', 'TS', 'SWC', 'LWdown', 'LW_OUT', 'H_CORR', 'Ts_model_1', 'Ts_model_2', 'Ts_model_3', 'SWC_model_1', 'SWC_model_2', 'SWC_model_3', 'SWC_model_4', 'evpg', 'evpa', 'Tleaf', 'mLwup', 'mLE', 'mH', 'mUstar', 'mfsno', 'mrss', 'LE', 'SH']].copy()
                print('LWUP NOT EXIST')
        else:
            df_new = data_swc[['Precip', 'Psurf', 'Tair', 'Qair', 'Ustar', 'Wind', 'LE_CORR', 'SWC_F_MDS_1_QC', 'TS_F_MDS_1_QC', 'LAI_N', 'SAI_N', 'year_qc', 'TS', 'SWC', 'LWdown', 'H_CORR', 'Ts_model_1', 'Ts_model_2', 'Ts_model_3', 'SWC_model_1', 'SWC_model_2', 'SWC_model_3', 'SWC_model_4', 'evpg', 'evpa', 'Tleaf', 'mLwup', 'mLE', 'mH', 'mUstar', 'mfsno', 'mrss', 'LE', 'SH']].copy()
        df_clean = df_new
        # Attach site constants and units to the per-site NetCDF output.
        ds = xr.Dataset.from_dataframe(df_clean)
        ds['ht'] = xr.DataArray(ht, attrs={'units': 'm'})
        ds['psi0'] = xr.DataArray(psi0, attrs={'units': 'mm'})
        ds['bsw'] = xr.DataArray(bsw)
        ds['porsl'] = xr.DataArray(porsl)
        ds['lamda'] = xr.DataArray(Lamda, attrs={'units': 'J/kg'})
        ds['hksati'] = xr.DataArray(hksati, attrs={'units': 'mm/s'})
        ds['om'] = xr.DataArray(om)
        ds['sand'] = xr.DataArray(sand)
        ds['TS_height'] = xr.DataArray(TS_height, attrs={'units': 'm'})
        ds['SWC_height'] = xr.DataArray(SWC_height, attrs={'units': 'm'})
        ds['Hcan'] = xr.DataArray(Hcan, attrs={'units': 'm'})
        ds['IGBP'] = xr.DataArray(IGBP)
        ds['lat'] = xr.DataArray(lat)
        ds['lon'] = xr.DataArray(lon)
        nc_filename = OUTPUT_DIR / BASE_FILE_PATTERN.format(sitename=sitename, period=common_parts[0])
        ds.to_netcdf(path=nc_filename, mode='w')
        print('----------------------------')
        print(sitename)
        print(f'Data saved to {nc_filename}')


if __name__ == "__main__":
    main()
