def energy(channel1, Det_Calib, Det_Offset, q_dept, k_dept, k_sigm, q_sigm, counts1, solid_angle, charge, Fluence):
    i1 = []
    i2 = []
    i3 = []
    i4 = []
    i5 = []
    i6 = []
    for i in range(len(channel1)):
        ener = Det_Calib*channel1[i] + Det_Offset
        dept = (ener - q_dept)/k_dept
        sigm = dept*k_sigm + q_sigm
        '''area density'''
        arde = counts1[i]/(sigm*solid_angle/1000*1e-24*charge*0.000001/1.602e-19)
        '''volume density'''
        vode = arde/((1/k_dept*Det_Calib*(-1))*0.0000001)
        '''normalized volume'''
        novo = vode/Fluence
        i1.append(ener)
        i2.append(dept)
        i3.append(sigm)
        i4.append(arde)
        i5.append(vode)
        i6.append(novo)
    return i1, i2, i3, i4, i5, i6
