import numpy as np
from math import pi, cos, radians
import utils as u

C = 2 * pi * 6378 * 6378 * 1000 * 1000

class region_class:
  def __init__(self, lat_a, lat_b,direction):
    self.lat_a = lat_a
    self.lat_b = lat_b
    self.direction = direction

class region_changed_class:
  def __init__(self, lat_a, lat_b,positive_change):
    self.lat_a = lat_a
    self.lat_b = lat_b
    self.positive_change = positive_change

def special_der(a0, a1, b0, b1):
    x = 1 / 6 * (a1 * b0 + a0 * b1) + 1 / 3 * (a0 * b0 + a1 * b1)
    return x



def reproduce(T0,changes):
    T_rep = np.array([T0])
    for i in range(0, len(changes)):
        T_rep = np.append(T_rep, T_rep[i] + changes[i])

    return T_rep






def Decompose(time,T,geopot,w,v,z,lat,lev):
    '''
    Data imput:  annual zonal mean data interpolated in same grid (time, lev, lat)
    T      # temperature in K
    geopot # geopotential height in gpm
    w      # vertical residual velocity in m/s
    v      # meridional residual velocity in m/s
    ztp    # geopotential height of particular level in gpm
    rho    # density in kg/m3
    lat    # latitude in radians, ordered increasingly
    lev    # decreasing pressure levels (eg. 1000,850,...)
    time   # time in years
    '''
    time_dim = len(time)
    lat_dim = len(lat)
    rho = u.CalculateDensity(lev,T,R=287.05)
    rhow = rho * w
    rhov = rho * v

    original_transport = np.zeros((time_dim))
    reconstructed_transport = np.zeros((time_dim))
    w_term = np.zeros((time_dim))
    v_term = np.zeros((time_dim))
    rho_term = np.zeros((time_dim))
    z_term = np.zeros((time_dim))
    shape_term = np.zeros((time_dim))
    width_term = np.zeros((time_dim))

    slopes = u.GetSlopes(z,lat_dim,time_dim) # get slope of the investigated level

    w_term_grid = np.zeros((time_dim -1, lat_dim))
    v_term_grid = np.zeros((time_dim -1, lat_dim))
    z_term_grid = np.zeros((time_dim -1, lat_dim))
    rho_term_grid = np.zeros((time_dim -1, lat_dim))
    shape_term_grid = np.zeros((time_dim -1, lat_dim))

    w_term_integrated = np.zeros((time_dim -1))
    v_term_integrated = np.zeros((time_dim -1))
    z_term_integrated = np.zeros((time_dim -1))
    rho_term_integrated = np.zeros((time_dim -1))
    shape_term_integrated = np.zeros((time_dim -1))
    width_term_integrated = np.zeros((time_dim -1))

    mass_fluxes, transport_signs = u.CalculateMassFlux(z, geopot, rhow, rhov, slopes, lat_dim, time_dim)


    for time_index in range(0, time_dim - 1):

                for lat_index in range(lat_dim):

                    z1 = z[time_index, lat_index]
                    z2 = z[time_index + 1, lat_index]

                    rho1 = np.interp(z1, geopot[time_index, :, lat_index], rho[time_index, :, lat_index])
                    rho2 = np.interp(z1, geopot[time_index + 1, :, lat_index], rho[time_index + 1, :, lat_index])
                    w1 = np.interp(z1, geopot[time_index, :, lat_index], w[time_index, :, lat_index])
                    w2 = np.interp(z1, geopot[time_index + 1, :, lat_index], w[time_index + 1, :, lat_index])
                    v1 = np.interp(z1, geopot[time_index, :, lat_index], v[time_index, :, lat_index])
                    v2 = np.interp(z1, geopot[time_index + 1, :, lat_index], v[time_index + 1, :, lat_index])

                    slope1 = slopes[time_index, lat_index]
                    slope2 = slopes[time_index + 1, lat_index]

                    w_term_grid[time_index, lat_index] = 0.5 * (rho1 + rho2) * (w2 - w1)
                    v_term_grid[time_index, lat_index] = (v2 - v1) * special_der(rho1, rho2, np.tan(slope1), np.tan(slope2))
                    shape_term_grid[time_index, lat_index] = (np.tan(slope2) - np.tan(slope1)) * special_der(v1, v2, rho1, rho2)

                    rho_term_grid[time_index, lat_index] = 0.5 * (w1 + w2) * (rho2 - rho1)
                    rho_term_grid[time_index, lat_index] += (rho2 - rho1) * special_der(v1, v2, np.tan(slope1), np.tan(slope2))

                    rho1 = np.interp(z1, geopot[time_index + 1, :, lat_index], rho[time_index + 1, :, lat_index])
                    rho2 = np.interp(z2, geopot[time_index + 1, :, lat_index], rho[time_index + 1, :, lat_index])
                    w1 = np.interp(z1, geopot[time_index + 1, :, lat_index], w[time_index + 1, :, lat_index])
                    w2 = np.interp(z2, geopot[time_index + 1, :, lat_index], w[time_index + 1, :, lat_index])
                    v1 = np.interp(z1, geopot[time_index + 1, :, lat_index], v[time_index + 1, :, lat_index])
                    v2 = np.interp(z2, geopot[time_index + 1, :, lat_index], v[time_index + 1, :, lat_index])

                    z_term_grid[time_index, lat_index] = rho2 * w2 - rho1 * w1 + rho2 * v2 * np.tan(slope2) - rho1 * v1 * np.tan(slope2)

    for i in range(0, time_dim - 1):

        region_i = region_class(u.GetRegions(mass_fluxes[i,:],transport_signs[i,:],lat,lat_dim))

        for region_index in range(len(region_i.lat_a)):

            intercept_a = region_i.lat_a[region_index]
            intercept_b = region_i.lat_b[region_index]

            if region_i.direction[region_index] == 1:

                w_term_integrated[i] += u.Integrate(w_term_grid[i, :], intercept_a, intercept_b, lat)
                v_term_integrated[i] += u.Integrate(v_term_grid[i, :], intercept_a, intercept_b, lat)
                z_term_integrated[i] += u.Integrate(z_term_grid[i, :], intercept_a, intercept_b, lat)
                rho_term_integrated[i] += u.Integrate(rho_term_grid[i, :], intercept_a, intercept_b, lat)
                shape_term_integrated[i] += u.Integrate(shape_term_grid[i, :], intercept_a, intercept_b, lat)

        region_i2 = region_class(u.GetRegions(mass_fluxes[i+1,:],transport_signs[i+1,:],lat,lat_dim))
        region_diff_act = region_changed_class(u.RegionDifference(region_i, region_i2,lat,lat_dim))

        for region_index in range(len(region_diff_act.lat_a)):

            intercept_a = region_diff_act.lat_a[region_index]
            intercept_b = region_diff_act.lat_b[region_index]

            if region_diff_act.positive_change[region_index]  == False:

                w_term_integrated[i] -= u.Integrate(w_term_grid[i, :], intercept_a, intercept_b, lat)
                v_term_integrated[i] -= u.Integrate(v_term_grid[i, :], intercept_a, intercept_b, lat)
                z_term_integrated[i] -= u.Integrate(z_term_grid[i, :], intercept_a, intercept_b, lat)
                rho_term_integrated[i] -= u.Integrate(rho_term_grid[i, :], intercept_a, intercept_b, lat)
                shape_term_integrated[i] -= u.Integrate(shape_term_grid[i, :], intercept_a, intercept_b, lat)
                width_term_integrated[i] -= u.Integrate(mass_fluxes[i, :], intercept_a, intercept_b, lat)

            else:

                width_term_integrated[i] += u.Integrate(mass_fluxes[i, :], intercept_a, intercept_b, lat)

        for i in range(0, time_dim):

            region_i = region_class(u.GetRegions(mass_fluxes[i, :], transport_signs[i, :], lat, lat_dim))

            for region_index in range(len(region_i.lat_a)):

                intercept_a = region_i.lat_a[region_index]
                intercept_b = region_i.lat_b[region_index]

                if region_i.direction[region_index] == 1:
                    original_transport[i] += u.Integrate(mass_fluxes[i, :], intercept_a, intercept_b, lat)



    original_transport = C*original_transport
    T_0 = original_transport[0]

    z_term = C * reproduce(T_0,z_term_integrated)
    shape_term = C * reproduce(T_0,shape_term_integrated)
    w_term = C * reproduce(T_0,w_term_integrated)
    v_term = C * reproduce(T_0,v_term_integrated)
    rho_term = C * reproduce(T_0,rho_term_integrated)
    width_term = C * reproduce(T_0,width_term_integrated)

print()