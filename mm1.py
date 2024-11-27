import numpy as np
import matplotlib.pyplot as plt

# Given temperature data
t_C = np.array([13, 14, 17, 18, 19, 15, 13, 31, 32, 29, 27])  # Celsius temperatures
t_F = np.array([55, 58, 63, 65, 66, 59, 56, 87, 90, 85, 81])  # Fahrenheit temperatures

# Number of parameters to estimate (k1 and k2)
NA = 2
x_r = np.zeros((NA, 1))  # Initial parameter estimates

# Function for online RLSE updates
def rlse_online(aT_k, b_k, x_r, P):
    aT_k = aT_k.reshape(1, -1)  # Reshape to 1D row vector
    K = P @ aT_k.T / (aT_k @ P @ aT_k.T + 1)  # Gain matrix (Eq. 2.1.17)
    x_r = x_r + K * (b_k - aT_k @ x_r)  # Update parameter estimates (Eq. 2.1.16)
    P = P - K @ aT_k @ P  # Update covariance matrix (Eq. 2.1.18)
    return x_r, K, P

# Initialize the loop parameters
g = 10  # Initial gain value
gain = []
x_f = np.zeros((10, 2))  # Store final estimates for each loop
P = g * np.eye(NA)  # Initial covariance matrix

# Perform the RLSE estimation with different gain matrix initializations
for i in range(10):
    P = (i + 1) * g * np.eye(NA)  # Update the initial covariance matrix based on i
    x_r = np.zeros((NA, 1))  # Reset initial parameter estimates

    # Loop over all data points for online RLSE
    for k in range(len(t_F)):
        A_r = np.array([t_F[k], 1])  # Construct the current data point row [t_F(k), 1]
        B_r = t_C[k]  # Corresponding target value (Celsius temperature)
        
        # Update parameters using online RLSE
        x_r, K, P = rlse_online(A_r, B_r, x_r, P)
    
    # Store the results
    print(f"With the gain matrix initial value {(i + 1) * g}, we get estimation values {x_r[0, 0]:.4f} and {x_r[1, 0]:.4f}")
    x_f[i, :] = x_r.flatten()  # Store the estimated values
    gain.append((i + 1) * g)

# Print values obtained via PINV (Moore-Penrose pseudoinverse)
A = np.vstack([t_F, np.ones(len(t_F))]).T  # Design matrix [t_F, 1]
x_pinv, _, _, _ = np.linalg.lstsq(A, t_C, rcond=None)
print(f"Values obtained via PINV: {x_pinv[0]:.4f} and {x_pinv[1]:.4f}")

# Plot the estimates of k1 and k2 as a function of the gain matrix initialization
plt.figure(figsize=(12, 6))

# Plot the k1 estimates
plt.subplot(1, 2, 1)
plt.plot(gain, x_f[:, 0], 'bo-', label='k1 from RLSE')
plt.axhline(y=x_pinv[0], color='r', linestyle='--', label='k1 from PINV')
plt.xlabel('Gain Matrix Initial Value')
plt.ylabel('k1 Estimate')
plt.title('k1 Estimate vs Gain Matrix')
plt.legend()
plt.grid(True)

# Plot the k2 estimates
plt.subplot(1, 2, 2)
plt.plot(gain, x_f[:, 1], 'go-', label='k2 from RLSE')
plt.axhline(y=x_pinv[1], color='r', linestyle='--', label='k2 from PINV')
plt.xlabel('Gain Matrix Initial Value')
plt.ylabel('k2 Estimate')
plt.title('k2 Estimate vs Gain Matrix')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Generate a smooth line for plotting
tf = np.linspace(55, 90, 100)

# Plot the fitted line for the last RLSE estimate and PINV estimate
tc_rlse = x_f[-1, 0] * tf + x_f[-1, 1]
tc_pinv = x_pinv[0] * tf + x_pinv[1]

plt.figure(figsize=(10, 6))
plt.plot(t_F, t_C, 'ro', label='Measured Data')  # Original data points
plt.plot(tf, tc_rlse, 'b-', label='RLSE Fitted Line (Final Estimate)', linewidth=2)  # RLSE fitted line
plt.plot(tf, tc_pinv, 'g--', label='Least-Squares Fitted Line', linewidth=2)  # Least-squares fitted line

# Add labels, title, and legend
plt.xlabel('Temperature in Fahrenheit (°F)')
plt.ylabel('Temperature in Celsius (°C)')
plt.title('Comparison of RLSE and Least-Squares Solutions')
plt.legend()
plt.grid(True)
plt.show()
