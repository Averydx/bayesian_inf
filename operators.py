import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate


###Constants

DOMAIN_SIZE = 1.0
N_POINTS = 50
N_TIME_STEPS = 100
TIME_STEP_LENGTH = 0.1

KINEMATIC_VISCOSITY = 0.0001

MAX_ITER_CG = None

element_length = DOMAIN_SIZE / (N_POINTS - 1)
scalar_shape = (N_POINTS, N_POINTS)
scalar_dof = N_POINTS**2
vector_shape = (N_POINTS, N_POINTS, 2)
vector_dof = N_POINTS**2 * 2
sample_res_shape = (20,20)

x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

# Using "ij" indexing makes the differential operators more logical. Take
# care when plotting.
X, Y = np.meshgrid(x, y, indexing="ij")

coordinates = np.concatenate(
    (
        X[..., np.newaxis],
        Y[..., np.newaxis],
    ),
    axis=-1,
)

###Operators

def partial_derivative_x(field):

        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[2:  , 1:-1]
                -
                field[0:-2, 1:-1]
            ) / (
                2 * element_length
            )
        )

        return diff

def partial_derivative_y(field):

    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = (
        (
            field[1:-1, 2:  ]
            -
            field[1:-1, 0:-2]
        ) / (
            2 * element_length
        )
    )

    return diff

def laplace(field):
    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = (
        (
            field[0:-2, 1:-1]
            +
            field[1:-1, 0:-2]
            - 4 *
            field[1:-1, 1:-1]
            +
            field[2:  , 1:-1]
            +
            field[1:-1, 2:  ]
        ) / (
            element_length**2
        )
    )

    return diff

def divergence(vector_field):
    divergence_applied = (
        partial_derivative_x(vector_field[..., 0])
        +
        partial_derivative_y(vector_field[..., 1])
    )

    return divergence_applied

def gradient(field):
    gradient_applied = np.concatenate(
        (
            partial_derivative_x(field)[..., np.newaxis],
            partial_derivative_y(field)[..., np.newaxis],
        ),
        axis=-1,
    )

    return gradient_applied

def curl_2d(vector_field):
    curl_applied = (
        partial_derivative_x(vector_field[..., 1])
        -
        partial_derivative_y(vector_field[..., 0])
    )

    return curl_applied

def advect(field, vector_field):
    backtraced_positions = np.clip(
        (
            coordinates
            -
            TIME_STEP_LENGTH
            *
            vector_field
        ),
        0.0,
        DOMAIN_SIZE,
    )

    advected_field = interpolate.interpn(
        points=(x, y),
        values=field,
        xi=backtraced_positions,
    )

    return advected_field

def diffusion_operator(vector_field_flattened):
    vector_field = vector_field_flattened.reshape(vector_shape)

    diffusion_applied = (
        vector_field
        -
        KINEMATIC_VISCOSITY
        *
        TIME_STEP_LENGTH
        *
        laplace(vector_field)
    )

    return diffusion_applied.flatten()

def poisson_operator(field_flattened):
    field = field_flattened.reshape(scalar_shape)

    poisson_applied = laplace(field)

    return poisson_applied.flatten()

def curl_2d(vector_field):
    curl_applied = (
        partial_derivative_x(vector_field[..., 1])
        -
        partial_derivative_y(vector_field[..., 0])
    )

    return curl_applied

## Project onto the observation space 
xs_sample = np.linspace(0, N_POINTS-1, sample_res_shape[0]).astype(int)
ys_sample = np.linspace(0, N_POINTS-1, sample_res_shape[1]).astype(int)

xs_sample_points = np.linspace(0, DOMAIN_SIZE, sample_res_shape[0])
ys_sample_points = np.linspace(0, DOMAIN_SIZE, sample_res_shape[1])

X_sample,Y_sample = np.meshgrid(xs_sample_points, ys_sample_points, indexing="ij")

#Construct the observation operator

# Total size
full_size = N_POINTS * N_POINTS * 2
obs_size = sample_res_shape[0] * sample_res_shape[1] * 2

# Observation operator
H = np.zeros((obs_size, full_size))

row = 0
for c in range(2):
    for j in ys_sample:
        for i in xs_sample: 
            col = c * N_POINTS * N_POINTS + j * N_POINTS + i
            H[row, col] = 1
            row += 1



def NS_STEP(velocities_prev,forcing,time_current,TIME_STEP_LENGTH):
    forces = forcing(
        time_current,
        coordinates,
    )


    '''Forcing must be pre-vectorized'''
    # (1) Apply Forces
    velocities_forces_applied = (
        velocities_prev
        +
        TIME_STEP_LENGTH
        *
        forces
    )

    # (2) Nonlinear convection (=self-advection)
    velocities_advected = advect(
        field=velocities_forces_applied,
        vector_field=velocities_forces_applied,
    )

    # (3) Diffuse
    velocities_diffused = splinalg.cg(
        A=splinalg.LinearOperator(
            shape=(vector_dof, vector_dof),
            matvec=diffusion_operator,
        ),
        b=velocities_advected.flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(vector_shape)

    # (4.1) Compute a pressure correction
    pressure = splinalg.cg(
        A=splinalg.LinearOperator(
            shape=(scalar_dof, scalar_dof),
            matvec=poisson_operator,
        ),
        b=divergence(velocities_diffused).flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(scalar_shape)

    # (4.2) Correct the velocities to be incompressible
    velocities_projected = (
        velocities_diffused
        -
        gradient(pressure)
        )
    
    return velocities_projected

def NS_STEP_WRONG(velocities_prev):

    # (2) Nonlinear convection (=self-advection)
    velocities_advected = advect(
        field=velocities_prev,
        vector_field=velocities_prev,
    )

    # (3) Diffuse
    velocities_diffused = splinalg.cg(
        A=splinalg.LinearOperator(
            shape=(vector_dof, vector_dof),
            matvec=diffusion_operator,
        ),
        b=velocities_advected.flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(vector_shape)

    # (4.1) Compute a pressure correction
    pressure = splinalg.cg(
        A=splinalg.LinearOperator(
            shape=(scalar_dof, scalar_dof),
            matvec=poisson_operator,
        ),
        b=divergence(velocities_diffused).flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(scalar_shape)

    # (4.2) Correct the velocities to be incompressible
    velocities_projected = (
        velocities_diffused
        -
        gradient(pressure)
        )
    
    return velocities_projected