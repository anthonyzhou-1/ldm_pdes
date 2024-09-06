# Datasets
## Dataloading
We use datamodules from Pytorch lightning to handle the dataset creation and dataloading. In particular, the datasets generally return a dictionary with the relevant values, controlled by flags in the config files. The basic keys that are generated are:
```
- x: shape (batch, n_t, (ny), nx, c). Spatiotemporal data with variable spatial dimensions
- pos: shape (batch, nx, d). Spatial positions of coordinates. Can include the time coordinate as well.
- prompt: shape (batch,). A text description of x.
```

Additionally, the datamodules store a normalizer object to scale and unscale input data. 

## Cylinder Flow 
- 1000/100 train/valid samples
- Incompressible NS in water, Re ~100-1000, dt = 0.01
- Around 2000 mesh points, downsampled to 25 timesteps
- Each data sample has a different shape, so they cannot be stacked. Therefore each data sample is in its own numbered dictionary ('0' has sample 0, '1' has sample 1, etc.). 
- Data Structure:
```
- dataset.h5 (keys: '0', '1', ... etc.)
    - '0' (keys: 'cells', 'mesh_pos', 'metadata', 'node_type', 'pressure', 'u', 'v')
        - 'cells': shape (num_edges, 3). Defines connectivity in triangular mesh. Only used for plotting
        - 'mesh_pos': shape (num_nodes, 2). Defines the position of each node in the mesh. 
        - 'node_type': shape (num_nodes, 1). Defines type of each node (0=fluid, 4=inlet, 5=outlet, 6=boundaries/walls)
        - 'pressure': shape (num_timesteps, num_nodes, 1). Defines pressure at each timestep for all mesh points.
        - 'u': shape (num_timesteps, num_nodes). Defines x-component of velocity at each timestep for all mesh points.
        - 'v': shape (num_timesteps, num_nodes). Defines y-component of velocity at each timestep for all mesh points.
        - 'metadata': (keys: 'center', 'domain_x', 'domain_y', 'prompt', 'radius', 'reynolds_number', 't_end', 'u_inlet', 'v_inlet')
            - 'center': shape (2,). Extracted center of cylinder, in meters.
            - 'domain_x': shape (2,). Bounds of x in the domain, in meters.
            - 'domain_y': shape (2,). Bounds of y in the domain, in meters.
            - 'prompt': shape(). Procedurally generated prompt using template in paper. Read with ['prompt'].asstr()[()].
            - 'radius': shape (). Extracted radius if cylinder, in meters. 
            - 'reynolds_number': shape (). Extracted Reynolds number of simulation.
            - 't_end': shape (). Final time of simulation.
            - 'u_inlet': shape(). x-component of velocity at the inlet.
            - 'v_inlet': shape(). y-component of velocity at the inlet.
    - '1', '2', ... etc.
```
## Smoke Buoyancy (NS2D)
- 2496/608 train/valid samples.
    - Datasets are divided into separates files with 32 samples each. This results in 78 training files (78x32=2496) and 19 valid files (19x32=608)
- Smoke driven by a buoyant force, dt=1.5
- 128x128 spatial resolution, with 56 timesteps.
- Each file contains 32 samples for a given seed, with uniform shape. The text captions are not uniform, so they are stored in a numbered dictionary as well.
- Data Structure:
```
- dataset.h5 (keys: 'train' or 'valid')
    - 'train' (keys: 'buo_y', 'dt', 'dx', 'dy', 't', 'text_labels', 'u', 'vx', 'vy', 'x', 'y')
        - 'buo_y': shape (32,). Contains a scalar buoyancy factor for each sample.
        - 'dt', 'dx', 'dy': shape (32,). Contains a scalar dt, dx, or dy for each sample.
        - 't': shape (32, num_timesteps). Contains the time at each timestep for each sample.
        - 'vx': shape (32, num_timesteps, resolution_x, resolution_y). Contains the x-component of velocity at each nodal position, timestep, and sample. 
        - 'y': shape (32, num_timesteps, resolution_x, resolution_y). Contains the y-component of velocity at each nodal position, timestep, and sample. 
        - 'u': shape (32, num_timesteps, resolution_x, resolution_y). Contains the smoke density at each nodal position, timestep, and sample. 
        - 'x': shape (32, resolution_x). Contains the x-position for each position along the x-axis for each sample.
        - 'y': shape (32, resolution_y). Contains the y-position for each position along the y-axis for each sample.
        - 'text_labels' (keys: '0', '1', ..., '31')
            - '0': shape (). Contains the text caption for the 0-th sample. Read with ['text_labels']['0'].asstr()[()]
            - '1', '2', ... '31': Contains text cpation for the n-th sample.
```