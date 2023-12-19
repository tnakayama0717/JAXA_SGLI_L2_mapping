import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

FNAME = 'GC1SG1_201809050115H04610_L2SG_SSTDQ_3001.h5'
DNAME = '/Image_data/SST'

with h5py.File(FNAME, 'r') as file:
    # Read SST data
    Data0 = file[DNAME][:]
    
    # Read attributes
    Err_DN = file[DNAME].attrs['Error_DN']
    Min_DN = file[DNAME].attrs['Minimum_valid_DN']
    Max_DN = file[DNAME].attrs['Maximum_valid_DN']
    Slope  = file[DNAME].attrs['Slope']
    Offset = file[DNAME].attrs['Offset']

    # Data processing
    Data1 = Data0.astype(float)
    Data1[Data0 == Err_DN] = np.nan
    Data1[(Data0 <= Min_DN) | (Data0 >= Max_DN)] = np.nan
    Data1 = Slope * Data1 + Offset

    # Read QA_flag
    QA_flag = file['/Image_data/QA_flag'][:]
    possibly_cloudy = np.bitwise_and(QA_flag, 2**12, dtype=np.uint16)
    acceptable = np.bitwise_and(QA_flag, 2**13, dtype=np.uint16)
    good = np.bitwise_and(QA_flag, 2**14, dtype=np.uint16)

    reliable = np.logical_or.reduce([good, acceptable, possibly_cloudy])

    # Plotting
    plt.figure()
    plt.imshow(Data1, cmap='jet')
    plt.colorbar()
    plt.savefig("default.png", format="png", dpi=2000)
    plt.show()
    

    # Apply reliability mask
    Data1[~reliable] = np.nan

    # Plotting with reliability mask
    plt.figure()
    plt.imshow(Data1, cmap='jet')
    plt.colorbar()
    plt.savefig("applying_QAflag.png", format="png", dpi=2000)
    plt.show()
   

    # Read Latitude and Longitude
    Lat = file['/Geometry_data/Latitude'][:]
    Lat_r = float(file['/Geometry_data/Latitude'].attrs['Resampling_interval'])
    Lon = file['/Geometry_data/Longitude'][:]
    Lon_r = float(file['/Geometry_data/Longitude'].attrs['Resampling_interval'])

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(1, Lat_r * Lat.shape[1] + 1, Lat_r),
                       np.arange(1, Lat_r * Lat.shape[0] + 1, Lat_r))

    Xq, Yq = np.meshgrid(np.arange(1, Data0.shape[1] + 1),
                         np.arange(1, Data0.shape[0] + 1))

    # Interpolate Latitude and Longitude
    f_lat = griddata((X.flatten(), Y.flatten()), Lat.flatten(), (Xq, Yq), method='linear')
    f_lon = griddata((X.flatten(), Y.flatten()), Lon.flatten(), (Xq, Yq), method='linear')

    LLroi = {'Lat': f_lat,
             'Lon': f_lon}

    # Indexing
    IDX_X = slice(4600, 4900)
    IDX_Y = slice(1550, 1750)

    LLroi['Lat'] = LLroi['Lat'][IDX_X, IDX_Y]
    LLroi['Lon'] = LLroi['Lon'][IDX_X, IDX_Y]
    Data1 = Data1[IDX_X, IDX_Y]

    # ROI calculation
    DDeg = 10/4800
    ROI = {'LatLim': [np.min(LLroi['Lat']), np.max(LLroi['Lat'])],
           'LonLim': [np.min(LLroi['Lon']), np.max(LLroi['Lon'])]}

    Latg = np.arange(ROI['LatLim'][1], ROI['LatLim'][0] - DDeg, -DDeg)
    Long = np.arange(ROI['LonLim'][0], ROI['LonLim'][1] + DDeg, DDeg)

    LLg = np.meshgrid(Latg, Long, indexing='ij')

    # Scattered interpolation
    points = np.column_stack((LLroi['Lat'].flatten(), LLroi['Lon'].flatten()))
    values = Data1.flatten()

    grid_lat, grid_lon = np.meshgrid(Latg, Long, indexing='ij')
    grid_points = np.column_stack((grid_lat.flatten(), grid_lon.flatten()))

    Data2 = griddata(points, values, grid_points, method='linear').reshape(grid_lat.shape)

    # Plotting
    plt.figure()
    plt.imshow(Data1, vmin=24, vmax=31, cmap='jet')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("Tokyo_bay.png", format="png", dpi=2000)
    
    plt.figure()
    plt.pcolormesh(Long, Latg, Data2, vmin=24, vmax=31, cmap='jet')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("Tokyo_bay_mapping.png", format="png", dpi=2000)
    plt.show()
