'''Implementing the ETKF for Navier Stokes.'''

import numpy as np
from operators import *
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

def forcing_function(time, point):
    time_decay = np.maximum(
        2.0 - 0.5 * time,
        0.0,
    )

    forced_value = (
        time_decay
        *
        np.where(
            (
                (point[0] > 0.4)
                &
                (point[0] < 0.6)
                &
                (point[1] > 0.1)
                &
                (point[1] < 0.3)
            ),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        )
    )

    return forced_value

forcing_function_vectorized = np.vectorize(
        pyfunc=forcing_function,
        signature="(),(d)->(d)",
    )


N_ENSEMBLE_MEMBERS = 100

rng = np.random.default_rng(0)

sigma = 0.5

observations = np.load('NS_data.npy')
ref_vels = np.load('NS_ref.npy')

N_TIME_STEPS = observations.shape[0]

observations_flat = observations.reshape(N_TIME_STEPS,-1)
R_obs = 0.00001 * np.eye(observations_flat.shape[1])
state_std = 1.0

ONES = np.ones((N_ENSEMBLE_MEMBERS,1))
ENSEMBLE_SHAPE = (*vector_shape,N_ENSEMBLE_MEMBERS)
FLAT_ENSEMBLE_SHAPE = (N_POINTS * N_POINTS * 2,N_ENSEMBLE_MEMBERS)

ensemble = np.zeros(ENSEMBLE_SHAPE)

field = rng.normal(0.0,1.0,size = ensemble.shape)

field = gaussian_filter(field, sigma=(0, sigma, sigma, 0))

#Enforce the BCs
field[0,:,:,:] = 0.
field[N_POINTS-1,:,:,:] = 0.

field[:,0,:,:] = 0.
field[:,N_POINTS-1,:,:] = 0.

ensemble = field

##Flatten ensemble 
ensemble = ensemble.reshape(FLAT_ENSEMBLE_SHAPE)

###Print info
print(f"Number of ensemble members: {N_ENSEMBLE_MEMBERS}")
print(f"Observation shape: {observations.shape}")


errs = []
time_current = 0.0
for n in tqdm(range(N_TIME_STEPS)): 
    time_current += TIME_STEP_LENGTH
    if n > 0: 
        for i in range(N_ENSEMBLE_MEMBERS):
            ensemble[...,i] = NS_STEP(ensemble[...,i].reshape(vector_shape),forcing_function_vectorized,
                                  time_current=time_current,TIME_STEP_LENGTH=TIME_STEP_LENGTH).reshape(vector_dof) + rng.normal(scale = state_std * np.sqrt(TIME_STEP_LENGTH),size = vector_dof)
            

    m_prior = 1/N_ENSEMBLE_MEMBERS * ensemble @ ONES @ (ONES.T)

    #m_post = m_prior

    anomalies_prior = ensemble - m_prior

    projected_anomalies = (H @ anomalies_prior)
    R_obs_inv = np.linalg.pinv(R_obs)

    LAM,Q = np.linalg.eigh(np.eye(N_ENSEMBLE_MEMBERS) + 1/(N_ENSEMBLE_MEMBERS - 1) * projected_anomalies.T @ R_obs_inv @ projected_anomalies)

    LAM_inv = 1/LAM * np.eye(len(LAM))

    LAM_inv_sqrt = (LAM**-1/2) * np.eye(len(LAM))

    K = 1/(N_ENSEMBLE_MEMBERS - 1) * anomalies_prior @ (Q @ LAM_inv @ Q.T) @ (projected_anomalies).T @ R_obs_inv

    obs_current = observations_flat[n,:].reshape(-1,1)

    m_post = m_prior + K @ (obs_current @ ONES.T - H @ m_prior)

    anomalies_post = anomalies_prior @ (Q @ LAM_inv_sqrt @ Q.T)

    ensemble = m_post + anomalies_post

    ###Enforce BCs 
    ensemble = ensemble.reshape(ENSEMBLE_SHAPE)
    ensemble[0,:,:,:] = 0.
    ensemble[N_POINTS-1,:,:,:] = 0.

    ensemble[:,0,:,:] = 0.
    ensemble[:,N_POINTS-1,:,:] = 0.
    ensemble = ensemble.reshape(FLAT_ENSEMBLE_SHAPE)

    m_post_plot = m_post[...,0].reshape(vector_shape)

    err = np.sqrt(np.mean((m_post[...,0] - ref_vels[n])**2))

    errs.append(err)

    ref_vel_curr = ref_vels[n].reshape(vector_shape)

    # curl = curl_2d(m_post[...,0].reshape(vector_shape))
    # plt.contourf(
    # X,
    # Y,
    # curl,
    # cmap=cmr.redshift,
    # levels=100,
    #     )

#     plt.quiver(
#             X, 
#             Y,
#             m_post_plot[..., 0],
#             m_post_plot[..., 1],
#             color="blue",
#         )
    
#     plt.quiver(
#             X, 
#             Y,
#             ref_vel_curr[..., 0],
#             ref_vel_curr[..., 1],
#             color="red",
#         )

#     plt.draw()
#     plt.pause(0.0001)
#     plt.clf()

plt.plot(errs)
plt.show()







