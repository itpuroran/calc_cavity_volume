import math
import numpy as np
import networkx as nx

# phase 0 - vapour, phase 1 - liquid

# ---- main parameters --------
DUMPFILE_NAME = 'dump.txt'
NUMBER_OF_CELLS_1D = 40
R_SEP = 1
R_NEAREST_NEIGHBOURS = 1.66
NUM_NEIGHBOURS_LIQUID = 1
ENCODING = 'cp1252' # 'utf-8'
# full list of encoding: https://docs.python.org/3/library/codecs.html#standard-encodings
# -----------------------------

open('result_cells.txt', 'w', encoding=ENCODING).close()
with open('result_volumes.txt', 'w', encoding=ENCODING) as result_file_volume:
    result_file_volume.write('timestep number_of_cells volume\n')


def set_phase_pars(pars: np.array, box: dict) -> np.array:
    """
    define the phase of particles from local density
    phase=0 - vapour, phase=1 - liquid
    """
    for par_start in pars:
        neighs = []
        for par in pars:
            if par_start['id'] != par['id']:
                
                dist_x = min(
                    abs(par_start['x'] - par['x']),
                    abs(box['x_length'] - abs(par_start['x'] - par['x']))
                )

                if dist_x ** 2 <= R_NEAREST_NEIGHBOURS ** 2:

                    dist_y = min(
                        abs(par_start['y'] - par['y']),
                        abs(box['y_length'] - abs(par_start['y'] - par['y']))
                    )

                    if dist_y ** 2 <= R_NEAREST_NEIGHBOURS ** 2:

                        dist_z = min(
                            abs(par_start['z'] - par['z']),
                            abs(box['z_length'] - abs(par_start['z'] - par['z']))
                        )

                        if dist_z ** 2 <= R_NEAREST_NEIGHBOURS ** 2:

                            dist = math.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

                            if dist <= R_NEAREST_NEIGHBOURS:
                                neighs.append(par['id'])
                                if len(neighs) >= NUM_NEIGHBOURS_LIQUID:
                                    par_start['phase'] = 1
                                    break 

    return pars


def virtual_lattice(box: list, size: list) -> np.array :
    """
    return numpy structured array which contained data of virtual lattice nodes
    """
    types = []
    names = ['id', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'phase', 'in_clust']
    for name in names:
        if name in ['id', 'phase', 'in_clust']:
            types.append((name, 'i'))
        else:
            types.append((name, None))

    number_of_cells = NUMBER_OF_CELLS_1D ** 3
    lattice = np.zeros(number_of_cells, dtype=types)
    count = 0

    for x_scaled in range(NUMBER_OF_CELLS_1D):
        for y_scaled in range(NUMBER_OF_CELLS_1D):
            for z_scaled in range(NUMBER_OF_CELLS_1D):
                x_real = box['x_left'] + x_scaled * size[0]
                y_real = box['y_left'] + y_scaled * size[1]
                z_real = box['z_left'] + z_scaled * size[2]
                lattice[count] = (count, x_real, y_real, z_real, x_scaled, y_scaled, z_scaled, 0, 0)
                count += 1

    return lattice


def id_from_scaled_coord(x_scaled: int, y_scaled: int, z_scaled: int) -> int:
    """
    return the ordinal number of the virtual lattice node from scaled coordinates
    """
    return int(
        (x_scaled % NUMBER_OF_CELLS_1D) * NUMBER_OF_CELLS_1D ** 2 +\
        (y_scaled % NUMBER_OF_CELLS_1D) * NUMBER_OF_CELLS_1D +\
        (z_scaled % NUMBER_OF_CELLS_1D)
        )


def apply_pbc(x_scaled: int, y_scaled: int, z_scaled:int):
    """
    return scaled coordinates of cell taking into account periodic boundary conditions
    """
    scaled_coord = [x_scaled, y_scaled, z_scaled]
    for ind, val in enumerate(scaled_coord):
        if val < 0:
            scaled_coord[ind] = NUMBER_OF_CELLS_1D + val
        elif val >= NUMBER_OF_CELLS_1D:
            scaled_coord[ind] = val - NUMBER_OF_CELLS_1D
    return scaled_coord


def find_near_cells(x_real: float, y_real: float, z_real: float, box: list, size: list) -> list:
    """
    return id of cells which are located near the point (x_real, y_real, z_real)
    """
    ids_near_cells = []

    x_scaled_nearest = round((x_real - box['x_left']) // size[0])
    y_scaled_nearest = round((y_real - box['y_left']) // size[1])
    z_scaled_nearest = round((z_real - box['z_left']) // size[2])

    treshold = round(4 * (R_SEP / size[0]))

    for i in range(-treshold, treshold, 1):
        for j in range(-treshold, treshold, 1):
            for k in range(-treshold, treshold, 1):
                x_scaled_near = x_scaled_nearest - i
                y_scaled_near = y_scaled_nearest - j
                z_scaled_near = z_scaled_nearest - k
                x_scaled_near, y_scaled_near, z_scaled_near = apply_pbc(x_scaled_near, y_scaled_near, z_scaled_near)
                id_near = id_from_scaled_coord(x_scaled_near, y_scaled_near, z_scaled_near)
                ids_near_cells.append(id_near)
    return ids_near_cells


def set_phase_cells(cells: np.array, pars: np.array, box: list, cell_size: list) -> np.array:
    """
    change field 'phase' for cells
    """
    for par in pars:
        if par['phase'] == 1:
            nearest_cells = find_near_cells(par['x'], par['y'], par['z'], box, cell_size)
            for id in nearest_cells:
                cell = cells[id]
            #for cell in cells:
                if cell['phase'] != 1:
                    dist_x = min(
                        abs(par['x'] - cell['x']),
                        abs(box['x_length'] - abs(par['x'] - cell['x']))
                    )

                    if dist_x ** 2 <= R_SEP ** 2:
                        dist_y = min(
                            abs(par['y'] - cell['y']),
                            abs(box['y_length'] - abs(par['y'] - cell['y']))
                        )

                        if dist_y ** 2 <= R_SEP ** 2:
                            dist_z = min(
                                abs(par['z'] - cell['z']),
                                abs(box['z_length'] - abs(par['z'] - cell['z']))
                            )

                            if dist_z ** 2 <= R_SEP ** 2:
                                dist = dist_x ** 2 + dist_y ** 2 + dist_z ** 2
                                if dist <= R_SEP ** 2:
                                    cell['phase'] = 1
    return cells


def find_biggest_cavity(cells: np.array) -> set:
    """
    return cell's id which belong the biggest cavity
    """
    G = nx.Graph()
    for cell in cells:
        if cell['phase'] == 0:
            G.add_node(cell['id'])
    for id in list(G.nodes):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    x_n, y_n, z_n = cells[id]['xs'] + dx, cells[id]['ys'] + dy, cells[id]['zs'] + dz
                    x_scaled, y_scaled, z_scaled = apply_pbc(x_n, y_n, z_n)
                    id_n = id_from_scaled_coord(x_scaled, y_scaled, z_scaled)
                    if id_n in list(G.nodes):
                        G.add_edge(id, id_n)
    return max(nx.connected_components(G), key=len)


def set_belong_cavity(cells: np.array, cavity_cells: list) -> np.array:
    """
    change field 'in_clust' for cells
    """
    for id in cavity_cells:
        cells[id]['in_clust'] = 1
    return cells


with open(DUMPFILE_NAME, 'r', encoding=ENCODING) as dump_file:
    while True:

        result_file_cells = open('result_cells.txt', 'a', encoding=ENCODING)
        result_file_volume = open('result_volumes.txt', 'a', encoding=ENCODING)
        result_file_pars = open('result_particles.txt', 'a', encoding=ENCODING)
        line = dump_file.readline().strip()

        if not line:
            print('End Of File')
            break

        if line == 'ITEM: TIMESTEP':

            particles = []
            timestep = int(dump_file.readline())
            print(f'{timestep} timestep is processing')

            tmp = dump_file.readline() # ITEM: NUMBER OF ATOMS
            number_of_particles = int(dump_file.readline())

            tmp = dump_file.readline() # ITEM: BOX BOUNDS pp pp pp
            lx0, lxl = [float(i) for i in dump_file.readline().split(' ')]
            ly0, lyl = [float(i) for i in dump_file.readline().split(' ')]
            lz0, lzl = [float(i) for i in dump_file.readline().split(' ')]
            box_boundaries = {
                'x_left': lx0,
                'x_right': lxl,
                'x_length': lxl - lx0,
                'y_left': ly0,
                'y_right': lyl,
                'y_length': lyl - ly0,
                'z_left': lz0,
                'z_right': lzl,
                'z_length': lzl - lz0
            }

            fields_line_lst = dump_file.readline().strip().split(' ')
            fields = fields_line_lst[2:]
            fields.append('phase')

            cell_sizes = [
                box_boundaries['x_length'] / NUMBER_OF_CELLS_1D,
                box_boundaries['y_length'] / NUMBER_OF_CELLS_1D,
                box_boundaries['z_length'] / NUMBER_OF_CELLS_1D
            ]

            dtype = []
            for field in fields:
                if field in ['id', 'type', 'phase']:
                    dtype.append((field, 'i'))
                else:
                    dtype.append((field, None))

            particles = np.zeros(number_of_particles, dtype=dtype)
            for id in range(number_of_particles):
                particle_data = [float(i) for i in dump_file.readline().strip().split(' ')]
                particle_data.append(0) # add phase 0 - vapour
                particles[id] = tuple(particle_data)

        particles = set_phase_pars(pars=particles, box=box_boundaries)
        cells_data = virtual_lattice(box=box_boundaries, size=cell_sizes)
        cells_data = set_phase_cells(
            cells=cells_data,
            pars=particles,
            box=box_boundaries,
            cell_size=cell_sizes
        )
        cells_of_biggest_cavity = find_biggest_cavity(cells=cells_data)

        cells_data = set_belong_cavity(cells=cells_data, cavity_cells=cells_of_biggest_cavity)

        result_file_cells.write('ITEM: TIMESTEP\n')
        result_file_cells.write(f'{timestep}\n')
        result_file_cells.write('ITEM: NUMBER OF ATOMS\n')
        result_file_cells.write(f'{NUMBER_OF_CELLS_1D ** 3}\n')
        result_file_cells.write('ITEM: BOX BOUNDS pp pp pp\n')
        result_file_cells.write(f'{lx0} {lxl}\n')
        result_file_cells.write(f'{ly0} {lyl}\n')
        result_file_cells.write(f'{lz0} {lzl}\n')
        result_file_cells.write('ITEM: ATOMS id x y z phase in_cluster \n')
        for cell in cells_data:
            result_file_cells.write(f"{cell['id']} {cell['x']:3.4f} {cell['y']:3.4f}\
                                    {cell['z']:3.4f} {cell['phase']} {cell['in_clust']}\n")


        cell_volume = cell_sizes[0] * cell_sizes[1] * cell_sizes[2]
        biggest_cavity_volume = len(cells_of_biggest_cavity) * cell_volume
        result_file_volume.write(f'{timestep} {len(cells_of_biggest_cavity)}\
                                 {biggest_cavity_volume:3.4f}\n')

        result_file_cells.close()
        result_file_volume.close()


        result_file_pars.write('ITEM: TIMESTEP\n')
        result_file_pars.write(f'{timestep}\n')
        result_file_pars.write('ITEM: NUMBER OF ATOMS\n')
        result_file_pars.write(f'{number_of_particles}\n')
        result_file_pars.write('ITEM: BOX BOUNDS pp pp pp\n')
        result_file_pars.write(f'{lx0} {lxl}\n')
        result_file_pars.write(f'{ly0} {lyl}\n')
        result_file_pars.write(f'{lz0} {lzl}\n')
        result_file_pars.write('ITEM: ATOMS id type x y z vx vy vz phase \n')
        for par in particles:
            result_file_pars.write(f"{par['id']} {par['type']}\
                                   {par['x']} {par['y']} {par['z']} \
                                   {par['vx']} {par['vy']} {par['vz']}\
                                    {par['phase']} \n")
        result_file_pars.close()
