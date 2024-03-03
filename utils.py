import numpy as np
from math import pi,cos

def GetSlopes(z,lat_dim,time_dim):
    y = pi * 6378000 / lat_dim
    slopes = np.zeros((time_dim, lat_dim))
    slopes[:, 0] = np.arctan((z[:, 1] - z[:, 0]) / y)
    slopes[:, lat_dim - 1] = np.arctan((z[:, lat_dim - 1] - z[:, lat_dim - 2]) / y)

    for lat_index in range(1, lat_dim - 1):
        slopes[:, lat_index] = -np.arctan((z[:, lat_index + 1] - z[:, lat_index - 1]) / (2 * y))

    return slopes

def CalculateDensity(lev,temperature,R):
    density = lev[np.newaxis, :, np.newaxis] / (temperature * R)
    return density

def CalculateMassFlux(z,geopot,rhow,rhov,slopes,lat_dim,time_dim):

    transport_sign = np.zeros((time_dim, lat_dim))
    mass_fluxes = np.zeros((time_dim , lat_dim))
    for time_index in range(time_dim):

        for lat_index in range(lat_dim):
            z0 = z[time_index, lat_index]

            mass_fluxes[time_index, lat_index] = np.interp(z0, geopot[time_index, :, lat_index], rhow[time_index, :, lat_index])
            mass_fluxes[time_index, lat_index] += \
                np.interp(z0, geopot[time_index, :, lat_index],rhov[time_index, :, lat_index])* np.tan(slopes[time_index, lat_index])

            if mass_fluxes[time_index, lat_index] > 0:
                transport_sign[time_index, lat_index] = 1 # 1 for upwelling, 0 for downwelling

    return mass_fluxes,transport_sign

def Intercept(x1, x2, y1, y2):
    x0 = -y1 * (x2 - x1) / (y2 - y1) + x1
    return x0

def GetRegions(transport,transport_sign,lat,lat_dim):
    lat_a = np.array([])
    lat_b = np.array([])
    directions = np.array([])
    intercept_a = lat[0]
    direction = 0

    if transport_sign[ 0] == 1:
        direction = 1

    for lat_index in range(1, lat_dim - 1):

        if transport_sign[lat_index] != direction:
            intercept_b = Intercept(lat[lat_index - 1], lat[lat_index], transport[lat_index - 1], transport[ lat_index])
            lat_a = np.append(lat_a, intercept_a)
            lat_b = np.append(lat_b, intercept_b)
            directions = np.append(directions, direction)
            dir = transport_sign[ lat_index]
            intercept_a = intercept_b

    if transport_sign[lat_dim - 1] != direction:

        intercept_b = Intercept(lat[lat_index], lat[lat_index + 1], transport[lat_index], transport[lat_index +1])
        lat_a = np.append(lat_a, intercept_a)
        lat_b = np.append(lat_b, intercept_b)
        directions = np.append(directions, direction)
        direction = transport_sign[lat_dim - 1]
        intercept_a = intercept_b
        lat_a = np.append(lat_a, intercept_a)
        lat_b = np.append(lat_b, lat[lat_dim - 1])
        directions = np.append(directions, direction)
    else:
        lat_a = np.append(lat_a, intercept_a)
        lat_b = np.append(lat_b, lat[lat_dim - 1])
        directions = np.append(directions, direction)

    return lat_a,lat_b,directions

def RegionDifference(region1, region2,lat,lat_dim):
    lat_a = np.array([])
    lat_b = np.array([])
    change = np.array([])
    a_intercept = lat[0]
    b_intercept = lat[0]
    step1 = 0
    step2 = 0

    while b_intercept < lat[lat_dim - 1]:

        b_intercept = min(region2.lat_b[step2], region1.lat_b[step1])

        if region1.direction[step1] != region2.direction[step2]:

            if region2.direction[step2] == 1:

                lat_a = np.append(lat_a, a_intercept)
                lat_b = np.append(lat_b, b_intercept)
                change = np.append(change, True)   # True for change from downwelling to upwelling
            else:
                lat_a = np.append(lat_a, a_intercept)
                lat_b = np.append(lat_b, b_intercept)
                change = np.append(change, False)
        if region2.lat_b[step2] > region1.lat_b[step1]:

            step1 += 1

        elif region2.lat_b[step2] < region1.lat_b[step1]:
            step2 += 1
        else:
            step1 += 1
            step2 += 1

        a_intercept = b_intercept

    return lat_a,lat_b,change


def Integrate(arg, a, b, lat):
    n = int((b - a) * 1000)

    x = np.linspace(a, b, num=n)

    integral = 0
    for i in range(len(x) - 1):
        value1 = np.interp(x[i], lat, arg)
        value2 = np.interp(x[i + 1], lat, arg)

        integral += cos(0.5 * x[i] + 0.5 * x[i + 1]) * (x[i + 1] - x[i]) * (value1 + value2) / 2

    return integral