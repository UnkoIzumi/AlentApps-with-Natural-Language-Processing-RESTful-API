# Python 3 program for the
# haversine formula
import math

# Python 3 program for the
# haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2))
    rad = 6371 #jari jari bumi
    c = 2 * rad * math.asin(math.sqrt(a))
    return c


# Driver code
if __name__ == '__main__':
    lat1 = -7.2326322
    lon1 = 112.767113
    lat2 = -7.2326460
    lon2 = 112.767120
    
    print("")
    km_jarak = haversine(lat1, lon1, lat2, lon2)
    print("Jarak {:0.2f} KM".format(km_jarak))
    print("")
    m_jarak = km_jarak*1000
    print("Jarak {:0.2f} M".format(m_jarak))