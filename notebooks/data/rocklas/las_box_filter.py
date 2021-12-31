import laspy

las = laspy.read('granite_dells.las') 

for i in las.point_format:
    print(i.name)

# points with real local coordinates
las.x.scaled_array().min()
las.x.scaled_array().max()
las.y.scaled_array().min()
las.y.scaled_array().max()
las.z.scaled_array().min()
las.z.scaled_array().max()

# color value has type of uint16
print(las.red.max())
print(las.red.min())
print(las.green.max())
print(las.green.min())
print(las.blue.max())
print(las.blue.min())

def box_filter(las, x1, y1, x2, y2):
    xgood = (las.x >= x1) & (las.x < x2)
    ygood = (las.y >= y1) & (las.y < y2)
    good = xgood & ygood 
    found = (las.x[good], las.y[good], las.z[good], las.red[good], las.green[good], las.blue[good])
    return found
