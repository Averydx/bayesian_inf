import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt
from operators import *

# Optional
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


def main():

    rng = np.random.default_rng()

    observations = np.zeros((N_TIME_STEPS,sample_res_shape[0] * sample_res_shape[1] * 2))
    ref_vels = np.zeros((N_TIME_STEPS,vector_dof))

    forcing_function_vectorized = np.vectorize(
        pyfunc=forcing_function,
        signature="(),(d)->(d)",
    )

    velocities_prev = np.zeros(vector_shape)

    time_current = 0.0
    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH

        # Advance to next time step
        velocities_prev = NS_STEP(velocities_prev,forcing_function_vectorized,
                                  time_current=time_current,TIME_STEP_LENGTH=TIME_STEP_LENGTH)

        velocities_prev_flat = velocities_prev.flatten()
        ref_vels[i] = velocities_prev_flat
        observations[i] = H @ velocities_prev_flat

        curl = curl_2d(velocities_prev)
        plt.contourf(
            X,
            Y,
            curl,
            cmap=cmr.redshift,
            levels=100,
        )
        plt.quiver(
            X, 
            Y,
            velocities_prev[..., 0],
            velocities_prev[..., 1],
            color="dimgray",
        )

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    plt.show()

    np.save('NS_data.npy',observations)
    np.save('NS_ref.npy',ref_vels)

if __name__ == "__main__":
    main()