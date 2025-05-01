import numpy as np
import matplotlib.pyplot as plt

def ricker_wavelet(f, length, dt):
    t= np.arange(-length/2, (length-dt)/2, dt)
    y= (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2))*np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

def zoeppritz(theta, vp1, vs1, rho1, vp2, vs2, rho2):
    theta= np.radians(theta)
    dvp= vp2 - vp1
    dvs= vs2 - vs1
    drho= rho2 - rho1
    avg_vp= (vp1 + vp2)/2
    avg_vs= (vs1 + vs2)/2
    avg_rho= (rho1 + rho2)/2
    
    reflectivity= 0.5*(1 + np.tan(theta)**2)*(dvp/avg_vp) - 4*(np.sin(theta)**2)*(dvs/avg_vs) + 0.5*(1 - 4*(np.sin(theta)**2)*(vs1**2/vp1**2))*(drho/avg_rho)
    print(type(reflectivity))
    
    return reflectivity

def convolve(wavelet, reflectivity, num_samples, dt, interface_position):
    synthetic_seismograms = []
    for rc in reflectivity:
        r= np.zeros(num_samples)
        interface_index= int(interface_position/dt)
        r[interface_index]= rc
        synthetic_seismogrm = np.convolve(r, wavelet, mode='same')
        synthetic_seismograms.append(synthetic_seismogrm)
    return synthetic_seismograms

def batzle_wang(P, T, fluid, S=None, G=None, api=None):
    """
    Calculate the elastic properties of reservoir fluids using the
    Batzle & Wang [1992] equations.

    :param P: Pressure (MPa)
    :param T: Temperature {deg C)
    :param fluid: Fluid type to calculate: brine, gas, or oil
    :param S: Salinity (brine only, in ppm)
    :param G: Gas gravity (gas mode only, ratio of gas density to air density
              at 15.6C and atmospheric pressure)
    :param api: American Petroleum Insitute (API) oil gravity
    """

    if fluid == 'brine':
        S = S / (10**6)     # ppm to fraction of one
        w = np.array([
                      [1402.85,   1.524,     3.437e-3,  -1.197e-5],
                      [4.871,    -0.0111,    1.739e-4,  -1.628e-6],
                      [-0.04783,   2.747e-4, -2.135e-6,   1.237e-8],
                      [1.487e-4, -6.503e-7, -1.455e-8,   1.327e-10],
                      [-2.197e-7,  7.987e-10, 5.230e-11, -4.614e-13],
        ])

        rhow = (1 + (10**-6)*(-80*T - 3.3*(T**2) + 0.00175*(T**3) +
                489*P - 2*T*P + 0.016*(T**2)*P - (1.3e-5)*(T**3)*P -
                0.333*(P**2) - 0.002*T*(P**2)))

        rhob = rhow + S*(0.668 + 0.44*S + (10**-6)*(300*P - 2400*P*S +
                         T*(80 + 3*T - 3300*S - 13*P + 47*P*S)))

        Vw = 0
        for i in range(4):
            for j in range(3):
                Vw = Vw + w[i][j]*T**i*P**j

        Vb = (Vw + S*(1170 - 9.8*T + 0.055*T**2 - 8.5e-5*T**3 + 2.6*P -
              0.0029*T*P - 0.0476*P**2) + S**(3/2)*(780 - 10*P + 0.16*P**2) -
              1820*S**2)

        out = {'rho': rhob, 'Vp': Vb}

    elif fluid == 'oil':
        Rg = 2.03*G*(P*np.exp(0.02878*api - 0.00377*T))**1.205
        rho0 = 141.5 / (api + 131.5)
        B0 = 0.972 + 0.00038*(2.4*Rg*(G/rho0)**0.5 + T + 17.8)**(1.175)

        rho_r = (rho0/B0)*(1 + 0.001*Rg)**-1    # pseudo-density of oil
        rhog = (rho0 + 0.0012*G*Rg)/B0          # density of oil with gas
        rhop = (rhog + (0.00277*P -             # correct for pressure
                1.71e-7*P**3)*(rhog - 1.15)**2 + 3.49e-4*P)

        rho = rhop / (0.972 + 3.81e-4*(T + 17.78)**1.175)  # correct for temp
        Vp = 2096*(rho_r / (2.6 - rho_r))**0.5 - 3.7*T + 4.64*P + 0.0115*(
            4.12*(1.08/rho_r - 1)**0.5 -1)*T*P

        out = {'rho': rho, 'Vp': Vp}

    elif fluid == 'gas':
        Ta = T + 273.15                 # absolute temperature
        Pr = P / (4.892 - 0.4048*G)     # pseudo-pressure
        Tr = Ta / (94.72 + 170.75*G)    # pseudo-temperature

        R = 8.31441
        d = np.exp(-(0.45 + 8*(0.56 - 1/Tr)**2)*Pr**1.2/Tr)
        c = 0.109*(3.85 - Tr)**2
        b = 0.642*Tr - 0.007*Tr**4 - 0.52
        a = 0.03 + 0.00527*(3.5 - Tr)**3
        m = 1.2*(-(0.45 + 8*(0.56 - 1/Tr)**2)*Pr**0.2/Tr)
        y = (0.85 + 5.6/(Pr + 2) + 27.1/(Pr + 3.5)**2 -
             8.7*np.exp(-0.65*(Pr + 1)))
        f = c*d*m + a
        E = c*d
        Z = a*Pr + b + E

        rhog = (28.8*G*P) / (Z*R*Ta)
        Kg = P*y / (1 - Pr*f/Z)

        out = {'rho': rhog, 'Kg': Kg}
    else:
        out = None

    return(out)

def plot_seismogram(time, seismogram, title):
    plt.figure(figsize=(10, 6))
    for i, seismogram in enumerate(seismogram):
        offset= i*0.4  #spacing b/w seismograms
        plt.plot(seismogram + offset, time, label=f'{angles[i]}°')
        plt.gca().invert_yaxis()
    plt.xlabel('Amplitude')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Parameters
f= float(input("Enter the frequency of Ricker wavelet in Hz:\n"))
interface= float(input("Enter the position of the interface in seconds:\n")) #sec
temperature= float(input("Enter the value of temperature in the second layer in deg Celcius:\n"))
wavelet_length= 0.2  #length of the wavelet in seconds
dt = 0.001  #time step in seconds
p1= float(input("Enter the values of P-wave velocity in the 1st layer in m/s:\n"))
s1= float(input("Enter the value of S-wave velocity in the 1st layer in m/s:\n"))
rho1= float(input("Enter the value of density of the 1st layer matrix in kg/m^3:\n"))
# phi1= float(input("Enter the value of porosity of the 1st layer:\n"))
# pressure1= float(input("Enter the value of fluid pressure:\n"))
k_min= float(input("Enter the value of Bulk Modulus of the 2nd layer mineral in GPa:\n")) #mineral modulus
p2= float(input("Enter the values of P-wave velocity in the 2nd layer matrix in m/s:\n"))
s2= float(input("Enter the value of S-wave velocity in the 2nd layer in m/s:\n"))
rho2= float(input("Enter the value of density of the 2nd layer matrix in kg/m^3:\n"))
phi2= float(input("Enter the value of porosity of the 2nd layer:\n"))
k_mat= k_min*(1-phi2)
if phi2!=0:
    fluid_type2= str(input("Determine the type of fluid inside the 2nd layer (Brine, Gas, Oil):\n"))
    rhof2= batzle_wang(10, temperature, fluid_type2, 35000, 0.06955, 38)["rho"]
    rho2= rho2*(1 - phi2) + rhof2*phi2

    if fluid_type2=='gas':
        kf= batzle_wang(10, temperature, fluid_type2, 35000, 0.06955, 38)["Kg"]
        k2= k_mat + ((1 - k_mat/k_min)**2/(phi2/kf + (1-phi2)/k_min - k_mat/(k_min**2)))
        p2= np.sqrt(float(k2)/float(rho2))

    else:
        p2f= batzle_wang(10, temperature, fluid_type2, 35000, 0.55646, 38)["Vp"]
        p2= np.sqrt((1 - phi2)*(p2**2) + phi2*(p2f**2))         #approx

interface_depth= p1/interface
# fluid_pressure= 10000*interface_depth #SI units
num_samples= int(interface*2000)  #smoothness and centre

angles= np.linspace(0, 40, 11)

t, wavelet= ricker_wavelet(f, wavelet_length, dt)

reflectivity_array= zoeppritz(angles, p1, s1, rho1, p2, s2, rho2)

synthetic_seismogram = convolve(wavelet, reflectivity_array, num_samples, dt, interface)

time= np.arange(0, num_samples*dt, dt)

plot_seismogram(time, synthetic_seismogram, 'Synthetic Seismogram for a 2-Layer Model')