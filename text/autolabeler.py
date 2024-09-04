import h5py 
import numpy as np

def get_cylinder_pos_radius(sample):
    node_type = sample["node_type"][:]
    mesh_pos = sample["mesh_pos"][:]
    mask = node_type[:, 0] == 6
    pos_wall = mesh_pos[mask]
    pos_cylinder = [] 
    for i in range(len(pos_wall)):
        if pos_wall[i][1] != 0 and pos_wall[i][1] < .4:
            pos_cylinder.append(pos_wall[i])

    pos_cylinder = np.array(pos_cylinder)
    center = np.mean(pos_cylinder, axis=0)
    radius = np.linalg.norm(pos_cylinder[0] - center)
    radius = np.round(radius, 4)

    return center, radius

def get_inlet_velocity(sample):
    node_type = sample["node_type"][:]
    u = sample["u"][:]
    mask = node_type[:, 0] == 4
    u_inlet = u[0][mask]
    u_inlet = np.mean(u_inlet)
    v_inlet = 0

    return u_inlet, v_inlet

def get_reynolds_number(v, l):
    # assume v is in m/s and l is in cm
    l = l/100
    kinematic_viscosity = .0000010533
    reynolds_number = v*l/kinematic_viscosity
    
    reynolds_number = np.rint(reynolds_number/10) * 10 # round to nearest 10

    return int(reynolds_number)

def get_domain(sample):
    mesh_pos = sample["mesh_pos"][:]
    max_x = mesh_pos[:, 0].max()
    max_y = mesh_pos[:, 1].max()
    min_x = mesh_pos[:, 0].min()
    min_y = mesh_pos[:, 1].min()

    return (min_x, max_x), (min_y, max_y)

def get_prompt(cylinder_radius,
               cylinder_pos,
               domain_x = (0, 6),
               domain_y = (0, 2),
               inlet_velocity = (1, 0),
               reynolds_number = 100,
               time_end = 1):
    
    if reynolds_number < 200:
        flow_regime = "The flow is fully laminar."
    elif reynolds_number < 350:
        flow_regime = "The flow is transitioning in the wake."
    else:
        flow_regime = "The flow is turbulent."
    
    cylinder_radius = cylinder_radius * 100 # convert to cm
    prompt = f"Fluid passes over a cylinder with a radius of {cylinder_radius:.2f} and position: {cylinder_pos[0]:.2f}, {cylinder_pos[1]:.2f}. Fluid enters with a velocity of {inlet_velocity[0]:.2f}. The Reynolds number is {reynolds_number}. {flow_regime}"
    return prompt

path = "/home/cmu/anthony/data/deepmind/cylinder_flow/mesh/valid_downsampled_labeled.h5"
f = h5py.File(path, 'a')

max_Re = 0
min_Re = 1000000

for key in f.keys():
    sample = f[key]
    center, radius = get_cylinder_pos_radius(sample)
    u_inlet, v_inlet = get_inlet_velocity(sample)
    reynolds_number = get_reynolds_number(u_inlet, 2*radius)
    domain_x, domain_y = get_domain(sample)
    prompt = get_prompt(radius, center, domain_x, domain_y, (u_inlet, v_inlet), reynolds_number, 6)

    if reynolds_number > max_Re:
        max_Re = reynolds_number
    if reynolds_number < min_Re:
        min_Re = reynolds_number

    metadata = {}
    metadata["center"] = center
    metadata["radius"] = radius
    metadata["u_inlet"] = u_inlet
    metadata["v_inlet"] = v_inlet
    metadata["reynolds_number"] = reynolds_number
    metadata["domain_x"] = domain_x
    metadata["domain_y"] = domain_y
    metadata["t_end"] = 6 
    metadata["prompt"] = prompt
    
    if 'metadata' in sample:
        del sample['metadata']

    sample.create_group("metadata")
    for key, value in metadata.items():
        sample['metadata'].create_dataset(key, data=value)

print(f"Max Reynolds number: {max_Re}")
print(f"Min Reynolds number: {min_Re}")
print(prompt)

f.close()



