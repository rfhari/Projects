import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations
import time

matplotlib.style.use("seaborn-muted")
params = {
    'font.family':'sans-serif',
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.frameon': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
}
plt.rcParams.update(params)

## in the LAMMPS code, there is a hard-coded energy conversion parameters to convert the
## energy in the unit of "real"
## -> double unitConv = mass2Kg*(A2m*A2m/time2s/time2s)*J2Kcalmol; // energy conversion
## Here if we want to use the metal units, we need to convert the factor into the units of eV
EnergyConvert = 1.44e20/6.242e18 #* 1e6  # J2Kcalmol = 1.44e20, J2eV = 6.242e18, ps^2/fs^2 = 1e6
print(f"energy factor: {EnergyConvert**2:.3e}")
start_time = time.time()


def corr_func(data, corr_len):
    data_1 = data[:, 0:15:3].sum(axis=1)  #xaxis
    data_2 = data[:, 1:15:3].sum(axis=1)  #yaxis
    data_3 = data[:, 2:15:3].sum(axis=1)  #zaxis

    result_x = np.correlate(data_1, data_1, mode='full')
    result_y = np.correlate(data_2, data_2, mode='full')
    result_z = np.correlate(data_3, data_3, mode='full')

    r1_x = result_x[int((result_x.size-1)/2):]
    r1_y = result_y[int((result_y.size-1)/2):]
    r1_z = result_z[int((result_z.size-1)/2):]

    corr_len_divide = np.arange(data_1.shape[0], 0, -1)
    
    r1_x_norm = r1_x/corr_len_divide
    r1_y_norm = r1_y/corr_len_divide
    r1_z_norm = r1_z/corr_len_divide


    r2_x = r1_x_norm[:corr_len+1]
    r2_y = r1_y_norm[:corr_len+1]
    r2_z = r1_z_norm[:corr_len+1]
    
    return r2_x, r2_y, r2_z

#temp_range = np.asarray([50, 100, 150, 200, 250, 300, 350, 400])
#temp_range = np.asarray([50, 100, 150, 160, 170, 180, 190, 210, 450, 500])
#temp_range = np.asarray([310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450])
#temp_range = np.asarray([420,	440,	460,	480,	500,	520,	540,	560,	580,	600,	650,	700]) #210, 220, 230, 240, 260, 270, 280, 290, 310, 320, 330, 340, 360, 370, 380, 390])#, 140, 160, 180, 250, 350, 400, 300, 200])#460, 470, 480, 490, 500])

#temp_range = np.asarray([420,	440,	460,	480,	500,	520,	540,	560,	580,	600,	650,	700, 200, 220, 240, 260, 280])
#temp_range = np.asarray([120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 50, 80, 100])
temp_range = np.asarray([50, 70, 80, 100])

for temp in temp_range:    
    print("Currently running:", temp, "K")
    ev2J = 1.60218e-19
    J2Kcalmol = 1.44e20
    kCal2J = 4186.0/6.02214e23
    kB = 1.38e-23 #(J/K)
    T = temp #K
    Box_length =  5.46*4 
    p2s = 1e-12
    f2s = 1e-15
    A2m = 1e-10
    s = 2  # hard coded in the LAMMPS code, please check "never=5;" in the fix_rigid_nh.cpp
    V = Box_length*Box_length*Box_length*A2m**3
    deltaT = 0.001*p2s
    Flux_conv = (ev2J)**2 * (A2m/p2s)**2 #because we have dt in ps and flux in real units ie time in fs
    Unit_conv = Flux_conv/kB/T/T/V*deltaT*s

    corr_len = 50000
    corrfull_x = np.zeros([corr_len+1, 5])
    corrfull_y = np.zeros([corr_len+1, 5])
    corrfull_z = np.zeros([corr_len+1, 5])
    #file_path = "../../multiple_seeds_" + str(temp) + "K_original/"
    file_path = "../../contributions_analysis/"+str(temp)+"K/"

    flux1 = np.loadtxt(file_path + "flux_data_NVE1.txt")
    flux1[:, :6] = flux1[:, :6] * 1/ev2J/J2Kcalmol
    
    flux2 = np.loadtxt(file_path + "flux_data_NVE2.txt")
    flux2[:, :6] = flux2[:, :6] * 1/ev2J/J2Kcalmol
    
    flux3 = np.loadtxt(file_path + "flux_data_NVE3.txt")
    flux3[:, :6] = flux3[:, :6] * 1/ev2J/J2Kcalmol
    
    flux4 = np.loadtxt(file_path + "flux_data_NVE4.txt")
    flux4[:, :6] = flux4[:, :6] * 1/ev2J/J2Kcalmol
    
    flux5 = np.loadtxt(file_path + "flux_data_NVE5.txt")
    flux5[:, :6] = flux5[:, :6] * 1/ev2J/J2Kcalmol
    
    print("Temp:", temp, "end of check 1:", (time.time() - start_time))
    
    corrfull_x[:, 0], corrfull_y[:, 0], corrfull_z[:, 0] = corr_func(flux1, corr_len)
    
    corrfull_x[:, 1], corrfull_y[:, 1], corrfull_z[:, 1] = corr_func(flux2, corr_len)
    corrfull_x[:, 2], corrfull_y[:, 2], corrfull_z[:, 2] = corr_func(flux3, corr_len)
    
    corrfull_x[:, 3], corrfull_y[:, 3], corrfull_z[:, 3] = corr_func(flux4, corr_len)
    corrfull_x[:, 4], corrfull_y[:, 4], corrfull_z[:, 4] = corr_func(flux5, corr_len)
    
    print("Temp:", temp, "end of check 2:", (time.time() - start_time))
    print("Temp:", temp, "data after corr in x, y and z:", corrfull_x.shape, corrfull_y.shape, corrfull_z.shape)
    
    comb = combinations([0, 1, 2, 3, 4], 4)
    k_all, k_all_x, k_all_y, k_all_z = [], [], [], []
    corr_all, corr_all_x, corr_all_y, corr_all_z = [], [], [], []

    for i in list(comb):
        corrfull_x_test, corrfull_y_test, corrfull_z_test = corrfull_x[:, i], corrfull_y[:, i], corrfull_z[:, i]
        r2_x = np.mean(corrfull_x_test, axis=1)
        r2_y = np.mean(corrfull_y_test, axis=1)
        r2_z = np.mean(corrfull_z_test, axis=1)
        r2 = (r2_x + r2_y + r2_z)/3
        
        corr_all.append(r2)
        corr_all_x.append(r2_x)
        corr_all_y.append(r2_y)
        corr_all_z.append(r2_z)
        
        intg_x2 = (r2_x[:-1] + r2_x[1:])*(s/2)
        intg_y2 = (r2_y[:-1] + r2_y[1:])*(s/2)
        intg_z2 = (r2_z[:-1] + r2_z[1:])*(s/2)
        intg2 = (r2[:-1] + r2[1:])*(s/2)
        
        CSum_x2 = np.cumsum(intg_x2, axis=0) 
        CSum_y2 = np.cumsum(intg_y2, axis=0) 
        CSum_z2 = np.cumsum(intg_z2, axis=0) 
        CSum2 = np.cumsum(intg2, axis=0) 
        
        k_new_x2 = CSum_x2 * (Unit_conv/s)
        k_new_y2 = CSum_y2 * (Unit_conv/s)
        k_new_z2 = CSum_z2 * (Unit_conv/s)
        k_new2 = CSum2 * (Unit_conv/s)
        
        k_all.append(k_new2)
        k_all_x.append(k_new_x2)
        k_all_y.append(k_new_y2)
        k_all_z.append(k_new_z2)
    
    k_all = np.asarray(k_all)
    k_ = np.mean(k_all[:, 20000:], axis=1) 
    k_mean = np.mean(k_)
    k_std = np.std(k_)
    print("Temp:", temp, "k_mean:", k_mean, "k_std:", k_std)
    print("Temp:", temp, "k_all shape:", k_all.shape)
    
    corr_time = np.arange(1, k_new_x2.shape[0]+1) * (s*deltaT)
    fig1 = "k_across_seeds_" + str(T) + ".png"
    
    plt.figure()
    plt.plot(corr_time, k_all[0, :], label='seed 1')
    plt.plot(corr_time, k_all[1, :], label='seed 2')
    plt.plot(corr_time, k_all[2, :], label='seed 3')
    plt.plot(corr_time, k_all[3, :], label='seed 4')
    plt.plot(corr_time, k_all[4, :], label='seed 5')
    plt.plot(corr_time, np.mean(k_all, axis=0), label='avg')
    plt.legend()
    plt.ylabel("k across seeds(W/mK)")
    plt.xlabel("Correlation time (s)")
    plt.savefig(fig1, bbox_inches='tight')
    plt.show()
    
    fig2 = "k_across_direction_" + str(T) + ".png"
    plt.figure()
    plt.plot(corr_time, np.mean(k_all_x, axis=0), label="k_x")
    plt.plot(corr_time, np.mean(k_all_y, axis=0), label="k_y")
    plt.plot(corr_time, np.mean(k_all_z, axis=0), label="k_z")
    plt.plot(corr_time, np.mean(k_all, axis=0), label="k_avg")
    plt.legend()
    plt.ylabel("k across direction (W/mK)")
    plt.xlabel("Correlation time (s)")
    plt.savefig(fig2, bbox_inches='tight')
    plt.show()
   
    k_save = k_all
    arr_name = "k_all" + str(T) + "K.npy"
    np.save(arr_name, k_save) 

    corr_time1 = np.arange(0, k_new_x2.shape[0]+1) * (s*deltaT)
    fig3 = "HCACF_across_direction_" + str(T) + ".png"
    corr_all_x, corr_all_y, corr_all_z, corr_all = np.asarray(corr_all_x), np.asarray(corr_all_y), np.asarray(corr_all_z), np.asarray(corr_all)
    corr_all_x_norm, corr_all_y_norm, corr_all_z_norm = corr_all_x/corr_all_x[:, 0].reshape(-1, 1), corr_all_y/corr_all_y[:, 0].reshape(-1, 1), corr_all_z/corr_all_z[:, 0].reshape(-1, 1)
    corr_all_norm = corr_all/corr_all[:, 0].reshape(-1, 1)

    plt.figure()
    plt.plot(corr_time1, np.mean(corr_all_x_norm, axis=0), label="total_corr_x")
    plt.plot(corr_time1, np.mean(corr_all_y_norm, axis=0), label="total_corr_y")
    plt.plot(corr_time1, np.mean(corr_all_z_norm, axis=0), label="total_corr_z")
    plt.plot(corr_time1, np.mean(corr_all_norm, axis=0), label="total_corr")
    plt.ylabel("New HCACF across direction")
    plt.xlabel("Correlation time (s)")
    plt.savefig(fig3, bbox_inches='tight')
    plt.show()

    fig4 = "HCACF_across_seeds_" + str(T) + ".png"
    corr_all = np.asarray(corr_all)
    
    corr_save = corr_all_norm
    arr_name = "corr_mean" + str(T) + "K.npy"
    np.save(arr_name, corr_save)

    plt.figure()
    plt.plot(corr_time1, corr_all_norm[0, :], label="seed 1")
    plt.plot(corr_time1, corr_all_norm[1, :], label="seed 2")
    plt.plot(corr_time1, corr_all_norm[2, :], label="seed 3")
    plt.plot(corr_time1, corr_all_norm[3, :], label="seed 4")
    plt.plot(corr_time1, corr_all_norm[4, :], label="seed 5")
    plt.plot(corr_time1, np.mean(corr_all_norm, axis=0), label="avg")
    plt.ylabel("New HCACF across seeds")
    plt.xlabel("Correlation time (s)")
    plt.savefig(fig4, bbox_inches='tight')
    plt.show()
    
    print("Temp:", temp, "end of check 3:", (time.time() - start_time))
