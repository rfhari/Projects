import numpy as np
import time
import math 
import matplotlib.pyplot as plt

def corr_func(data, corr_len):
    result_x = np.correlate(data, data, mode='full')
    r1_x = result_x[int((result_x.size-1)/2):]
    corr_len_divide = np.arange(data.shape[0], 0, -1)    
    r1_x_norm = r1_x/corr_len_divide
    r2_x = r1_x_norm[:corr_len+1]
    return r2_x

def vector_racf(all_dir, corr_len):
    corr_vec = []
    first = 1
    for i in range(0, 256):
        curr_dir = all_dir[i, :, :]
        # print("curr_dir shape:", curr_dir.shape)
        temp_x = corr_func(curr_dir[:, 0], corr_len)
        temp_y = corr_func(curr_dir[:, 1], corr_len)
        temp_z = corr_func(curr_dir[:, 2], corr_len)
        if first == 1:
            temp_tot = temp_x + temp_y + temp_z
            corr_vec = temp_tot
            first = 0
        else:
            temp_tot = temp_x + temp_y + temp_z
            corr_vec = np.vstack ((corr_vec, temp_tot))
    return corr_vec

def angles_racf(angles, corr_len):
    corr_angles = []
    first = 1
    for i in range(0, 256):
        angle = angles[i, :]
        # print("angle:", angle.shape)
        temp = corr_func(angle, corr_len)
        # print("temp:", temp.shape)
        if first == 1:
            corr_angles = temp
            first = 0
        else:
            # print("corr_angles:", corr_angles.shape)
            corr_angles = np.vstack ((corr_angles, temp))
    # print("corr_angles:", corr_angles.shape)
    return corr_angles

def find_dir(value_matrix):
    direc = [] #[mol_id, dx, dy, dz]
    for i in range(0, value_matrix.shape[0], 2):
        mol_id = value_matrix[i, 1]  #mol_id
        if (value_matrix[i, 0] > value_matrix[i+1, 0]):
            dx = value_matrix[i, 2] - value_matrix[i+1, 2] 
            dy = value_matrix[i, 3] - value_matrix[i+1, 3]
            dz = value_matrix[i, 4] - value_matrix[i+1, 4]
        elif (value_matrix[i, 0] < value_matrix[i+1, 0]):
            dx = value_matrix[i+1, 2] - value_matrix[i, 2] 
            dy = value_matrix[i+1, 3] - value_matrix[i, 3]
            dz = value_matrix[i+1, 4] - value_matrix[i, 4]
        vec_len = np.sqrt(dx**2+dy**2+dz**2)
        if vec_len > 1.315 or vec_len < 1.305:
            print("vec_len:", vec_len, " molID:", mol_id)
            error_flag = 1
            break
        else:
            direc.append([mol_id, dx/vec_len, dy/vec_len, dz/vec_len])
    return np.asarray(direc)

#temp_range = np.asarray([190, 200, 210, 220, 230, 240, 250, 260, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450])
temp_range = np.asarray([170])

for Temp in temp_range:
    tot_atoms = 256*2
    box_len = 5.46*4
    time_index = 0
    flag = 0
    skip = 0
    first = 0
    count = 0
    flag_first_fill = 0
    Jxx, Jyy, Jzz, kallT, timesteps, ke = [], [], [], [], [], []
    atom_id, mol_id, atom_type, x, y, z = [], [], [], [], [], []
    start_time = time.time()
    oxy_len = 1.31
    print("Currently_running:", Temp)
    #file_path = "../multiple_seeds_"+str(Temp)+"K_original/Atoms_NaO2_NVT_1.dump"
    # file_path = "../contributions_analysis/"+str(Temp)+"K/Atoms_NaO2_NVE_1.dump"
    file_path = "./"+str(Temp)+"K/Atoms_NaO2_NVE_1.dump"

    for line in open(file_path):
        listWords = line.split(" ")
        item = listWords
        if (skip == 9 and first == 0):
            if (len(item) > 2 and float(item[2])==2):
              first = 1
              count = count + 1
              atom_id.append(float(item[0]))
              mol_id.append(float(item[1]))
              atom_type.append(float(item[2]))
              x.append(float(item[6]))
              y.append(float(item[7]))
              z.append(float(item[8]))
        elif (skip == 9 and first == 1):
            if (len(item) == 9 and float(item[2])==2):
              count = count + 1
              atom_id.append(float(item[0]))
              mol_id.append(float(item[1]))
              atom_type.append(float(item[2]))
              x.append(float(item[6]))
              y.append(float(item[7]))
              z.append(float(item[8]))
        elif (skip < 9):
            skip += 1    
        if (count == tot_atoms):
            values = np.asarray([atom_id, mol_id, x, y, z]).T
            sorted_values = np.asarray(sorted(values,key=lambda x: x[1]))
            if flag_first_fill == 0:
                initial_dir = find_dir(sorted_values) #[atom_id, mol_id, x, y, z]  
                flag_first_fill = 1
            else:
                break
            break
    
    end_time = time.time()      
    print("read first:", end_time - start_time, "count:", count)
    
    ts = 0
    flag = 0
    skip = 0
    first = 0
    count = 0
    flag_first_fill, special_one = 0, 0
    Jxx, Jyy, Jzz, kallT, timesteps, ke = [], [], [], [], [], []
    atom_id, mol_id, atom_type, x, y, z = [], [], [], [], [], []
    NaN_ts, NaN_sum = [], []
    
    for line in open(file_path):
        listWords = line.split(" ")
        item = listWords
        if (skip == 9 and first == 0):
            if (len(item) > 2 and float(item[2])==2):
              first = 1
              count = count + 1
              atom_id.append(float(item[0]))
              mol_id.append(float(item[1]))
              atom_type.append(float(item[2]))
              x.append(float(item[6]))
              y.append(float(item[7]))
              z.append(float(item[8]))
        elif (skip == 9 and first == 1):
            if (len(item) == 9 and float(item[2])==2):
              count = count + 1
              atom_id.append(float(item[0]))
              mol_id.append(float(item[1]))
              atom_type.append(float(item[2]))
              x.append(float(item[6]))
              y.append(float(item[7]))
              z.append(float(item[8]))
        elif (skip < 9):
            skip += 1    
        if (count == tot_atoms):
            values = np.asarray([atom_id, mol_id, x, y, z]).T
            sorted_values = np.asarray(sorted(values, key=lambda x: x[1]))
            if flag_first_fill == 0:
                # initial_dir = find_dir(sorted_values) #[atom_id, mol_id, x, y, z]  
                curr_dir = find_dir(sorted_values)
                all_dir = curr_dir[:, 1:].reshape(256, 1, 3)
                temp = np.multiply(curr_dir[:, 1:], initial_dir[:, 1:])
                temp = np.sum(temp, axis=1).reshape(-1, 1)
                temp_modified = [1.0 if vc <1.1 and vc >  0.99999 else vc[0] for vc in temp]
                # value_original = [[vc, i] for i, vc in enumerate(temp) if vc <1.1 and vc >  0.99999]
                # temp_modified = np.asarray(temp_modified).reshape(-1, 1)
                new_angles = np.arccos(temp_modified) * (180/np.pi)
                all_angles = new_angles.reshape(-1, 1)
                flag_first_fill = 1
            else:
                curr_dir = find_dir(sorted_values)
                temp = np.multiply(curr_dir[:, 1:], initial_dir[:, 1:])
                temp = np.sum(temp, axis=1).reshape(-1, 1)
                value_original = [[vc[0], i] for i, vc in enumerate(temp) if vc <1.1 and vc >  0.9999]
                value_exact = [vc[0] for i, vc in enumerate(temp) if vc == 1.0000]
                value_only = [vc[0] for i, vc in enumerate(temp) if vc <1.1 and vc >  0.9999]
                mol_only = [i for i, vc in enumerate(temp) if vc <1.1 and vc >  0.9999]
                new_angles = np.arccos(temp) * (180/np.pi)                
                new_angles = new_angles.reshape(-1, 1)
                NaN_sum.append(np.isnan(np.sum(new_angles)))
                all_angles = np.hstack([all_angles, new_angles])
                all_dir = np.hstack([all_dir, curr_dir[:, 1:].reshape(256, 1, 3)])
                if special_one == 0:
                    special_array = [np.asarray(value_original)]
                    tot_value_only = [value_only]
                    tot_mol_only = [mol_only]
                    exact_array = [value_exact]
                    special_one = 1
                else:
                    tot_value_only.append(value_only)
                    tot_mol_only.append(mol_only)
                    special_array.append(np.asarray(value_original))
                    exact_array.append(value_exact)              
            print("Temp:", Temp, "Time step:", ts)
            ts+=10
            count = 0 
            skip = 9
            atom_id, mol_id, atom_type, value_only, mol_only, value_original, value_exact = [], [], [], [], [], [], []
            x, y, z = [], [], []
    
    end_time = time.time()        
    print("time taken:", end_time - start_time, "sec")
    print("all_angles:", all_angles.shape)
    
    time_steps = np.arange(0, 1e6+1, 10)
    avg_particles = np.mean(all_angles, axis=0)
    fig_name2 = "avg_deviations_" + str(Temp) + "K.png"
    plt.figure()
    plt.plot(time_steps, avg_particles)
    plt.xlabel("Time steps")
    plt.ylabel("Avg. Degrees of deviations")
    plt.savefig(fig_name2)
    plt.show()

    np.save("all_angles_"+str(Temp)+"K_1.npy", all_angles)
    np.save("all_directions_"+str(Temp)+"K_1.npy", all_dir)
    
    corr_len = 60000
    s = 10 #100 for NVE and 10 for NVT
    p2s =  1e-12
    deltaT = 0.001*p2s
    f2s = 1e-15
    all_racf = vector_racf(all_dir, corr_len)
    corr_time = np.arange(0, all_racf.shape[1]) * (s*deltaT)
    tp = all_racf[:, 0].reshape(-1, 1)
    all_racf_norm = all_racf/tp
    nan_values = np.isnan(np.mean(all_racf_norm, axis=1)).sum()
    print("all_racf_norm:", all_racf_norm.shape)
    avg_racf_norm = np.mean(all_racf_norm, axis=0)
    fig_name = "RACF_" + str(Temp) + ".png"
    y_title = "RACF at " + str(Temp) + " K"
    if (Temp>120):
        plt.figure()
        plt.plot(corr_time/p2s, all_racf_norm.T, 'b-')
        plt.plot(corr_time/p2s, avg_racf_norm, 'g-')
        plt.ylabel(y_title)
        plt.xlabel("Correlation time (ps)")
        plt.ylim(-1, 1)
        plt.savefig(fig_name, dpi=500)
        plt.show()
    else:
        plt.figure()
        plt.plot(corr_time/p2s, all_racf_norm.T, 'b-')
        plt.plot(corr_time/p2s, avg_racf_norm, 'g-')
        plt.ylabel(y_title)
        plt.xlabel("Correlation time (ps)")
        plt.savefig(fig_name, dpi=500)
        plt.show()
    end_time = time.time()        
    print("time taken:", end_time - start_time, "sec")
    save_racf = "avg_racf_norm_"+str(Temp)+"_1.npy"
    # np.save(save_racf, avg_racf_norm)
    np.save("all_racf_norm_"+str(Temp)+"_1.npy", all_racf_norm)
