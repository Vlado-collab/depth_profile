
def get_table1(filename1):
    dataFile = open(filename1, 'r')
    depth1 = []
    energy1 = []
    sigma1 = []
    for line in dataFile:
        x1, y1, z1 = line.split()
        depth1.append(float(x1))
        energy1.append(float(y1))
        sigma1.append(float(z1))
    dataFile.close()
    return (depth1, energy1, sigma1)
