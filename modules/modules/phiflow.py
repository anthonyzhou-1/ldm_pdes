from phi.flow import (  # SoftGeometryMask,; Sphere,; batch,; tensor,
    Box,
    CenteredGrid,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
)
from phi.math import reshaped_native, spatial, channel
import phi 
import torch 
from tqdm import tqdm

def simulate_fluid(inputs, buoyancy_y):
    # inputs in shape b nt nx ny nc 
    nt = 56
    tmin = 18.0
    tmax = 102.0
    nx = ny = 128 
    Lx = Ly = 32
    nu = 0.01
    dt = (tmax - tmin) / nt

    # resimulate the fluid based on the inputs
    initial_smoke = inputs[0, 0, :, :, -1]
    initial_velocity = inputs[0, 0, :, :, :2]

    initial_smoke = phi.math.tensor(initial_smoke, spatial('x,y'))
    initial_velocity =  phi.math.tensor(initial_velocity, spatial('x,y'), channel('vector'))

    smoke = abs(
        CenteredGrid(
            initial_smoke,
            extrapolation.BOUNDARY,
            x=nx,
            y=ny,
            bounds=Box['x,y', 0 : Lx, 0 : Ly],
        )
    )  # sampled at cell centers
    velocity = StaggeredGrid(
        initial_velocity, extrapolation.ZERO, x=nx, y=ny, bounds=Box['x,y', 0 : Lx, 0 : Ly]
    )  # sampled in staggered form at face centers


    fluid_field_ = [] 
    velocity_ = []

    fluid_field_.append(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True))
    velocity_.append(
        reshaped_native(
            velocity.staggered_tensor(),
            groups=("x", "y", "vector"),
            to_numpy=True,
        )
    )


    for i in tqdm(range(0, nt-1), desc="Simulating Fluid", leave=False):
        smoke = advect.semi_lagrangian(smoke, velocity, dt)
        buoyancy_force = (smoke * (0, buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
        velocity = advect.semi_lagrangian(velocity, velocity, dt) + dt * buoyancy_force
        velocity = diffuse.explicit(velocity, nu, dt)
        velocity, _ = fluid.make_incompressible(velocity)
        fluid_field_.append(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True))
        velocity_.append(
            reshaped_native(
                velocity.staggered_tensor(),
                groups=("x", "y", "vector"),
                to_numpy=True,
            )
        )

    fluid_field_tensor = torch.tensor(fluid_field_)
    velocity_ = [v[:128, :128] for v in velocity_]
    velocity_tensor = torch.tensor(velocity_)

    fluid_field_tensor = fluid_field_tensor[:48]
    velocity_tensor = velocity_tensor[:48]

    output_tensor = torch.cat((velocity_tensor, fluid_field_tensor), dim=-1)
    output_tensor = output_tensor.unsqueeze(0)

    return output_tensor