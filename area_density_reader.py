def get_table4(filename1):
    dataFile = open(filename1, 'r')
    depth4 = []
    area_density4 = []
    for line in dataFile:
        x1, y1 = line.split()
        depth4.append(float(x1))
        area_density4.append(float(y1))
    dataFile.close()
    return (depth4, area_density4)
