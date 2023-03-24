import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import pandas as pd
import time
from datetime import datetime, timedelta


def func_timer(function_to_execute):
    """Measure the execution time of a function. It can be used as a decorator.

    Example:
        >>> @func_timer
        >>> def target_func():
        >>>     pass
    """
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        calc_time = round(timedelta(seconds=end_time -
                          start_time).total_seconds(), 5)
        print(
            f'[Info] "{function_to_execute.__name__}" is done in {calc_time} sec.')
        return result
    return compute_execution_time


def import_input(input_path, return_all=False):
    """import node & elem data from Nastran input file (.nas)

    Args:
        input_path (str): the file path of Nastran input file (.nas)
        return_all (bool, optional): If True, dictionaries are returned together with node_data and elem_data.

    Returns:
        array: node_data, elem_data, node_dict, node_idict, elem_dict, elem_idict
    """
    print_log('This function is deprecated. Please use "import_model" instead.')
    try:
        with open(input_path) as f:
            l = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, encoding='Shift-JIS') as f:
            l = f.readlines()

    # This function only captures shell elements.
    elem_lines = []
    node_lines = []
    shell = []
    for i in range(0, len(l)):
        if l[i][:6] == 'CQUAD4' or l[i][:6] == 'CTRIA3':
            elem_lines.append(i)
        elif l[i][:4] == 'GRID':
            node_lines.append(i)
        elif l[i][:6] == 'PSHELL':
            shell.append([int(l[i].split()[1]), float(l[i].split()[3])])
        elif l[i][:5] == 'PCOMP':
            shell.append([int(l[i].split()[1]), 2*abs(float(l[i].split()[2]))])

    shell = np.array(shell)
    shell_dict = dict(zip(shell[:, 0], shell[:, 1]))

    global elem_data
    elem_data = np.empty((len(elem_lines), 7))
    for i in range(0, len(elem_lines)):
        row = [0]*7
        pieces = l[elem_lines[i]].split()
        if pieces[0] == 'CTRIA3':
            row[0] = pieces[1]
            row[1:4] = pieces[3:6]
            row[4] = row[3]
            row[5] = shell_dict[int(pieces[2])]
            row[6] = 1
        elif pieces[0] == 'CQUAD4':
            row[0] = pieces[1]
            row[1:5] = pieces[3:7]
            row[5] = shell_dict[int(pieces[2])]
            row[6] = 1
        elem_data[i, :] = row
    elem_data = elem_data[np.argsort(elem_data[:, 0])]

    global node_data
    node_data = np.empty((len(node_lines), 4))
    for i in range(0, len(node_lines)):
        row = [0]*4
        pieces = l[node_lines[i]]
        row[0] = int(pieces[8:16])
        row[1] = float(pieces[24:32])
        try:
            row[2] = float(pieces[32:40])
        except ValueError:
            row[2] = 0
        try:
            row[3] = float(pieces[40:48])
        except ValueError:
            row[3] = 0
        node_data[i, :] = row

    # Nodes which are not included in shell definitions are removed.
    shell_node = np.unique(elem_data[:, 1:5].reshape((-1,)))
    node_remove = list(set(node_data[:, 0]) - set(shell_node))
    node_remove_index = []
    for i in range(0, node_data.shape[0]):
        if node_data[i, 0] in node_remove:
            node_remove_index.append(i)
    node_data = np.delete(node_data, node_remove_index, axis=0)

    # node & element identification numbers are assigned so that they are straight(not skipped) numbers.
    # node_dict & elem_dict is the dictionary that associates the original node number with the new node number.
    global node_id, node_dict, node_idict
    node_id = node_data[:, 0].copy()
    node_data[:, 0] = np.arange(1, len(node_data)+1)
    node_dict = dict(zip(node_id, node_data[:, 0]))
    node_idict = dict(zip(node_data[:, 0], node_id))
    for i in range(0, len(elem_data)):
        for j in range(1, 5):
            if np.isnan(elem_data[i, j]) == False:
                elem_data[i, j] = node_dict[elem_data[i, j]]
    global elem_id, elem_dict, elem_idict
    elem_id = elem_data[:, 0].copy()
    elem_data[:, 0] = np.arange(1, len(elem_data)+1)
    elem_dict = dict(zip(elem_id, elem_data[:, 0]))
    elem_idict = dict(zip(elem_data[:, 0], elem_id))

    if return_all:
        return node_data, elem_data, node_dict, node_idict, elem_dict, elem_idict
    else:
        return node_data, elem_data


def import_result(output_path, result='strain', coords_sys='local', composite=False, ply_num=2, ply_id=[]):
    """import displacement & strain data from Nastran output file(.OUT). Displacement and strain must be output in the output file.

    Args:
        output_path (str): the file path of Nastran output file(.OUT)
        result (str, optional): result quantity name (strain or stress).
        coords_sys (str, optional): 'local' or 'global'
        composite (bool, optional): True or False
        ply_num (int): The number of plies.
        ply_id (list, optional): ply id that is exported strain 

    Returns:
        array: strain_data, disp_data
    """
    assert result in (
        'strain', 'stress'), f'quantity name "{result}" is not defined.'
    k = 2 if result == 'strain' else 1

    result_data = np.empty((1, 19))
    if len(ply_id) != 0:
        ply_id = [str(_) for _ in ply_id]
    with open(output_path, 'r') as f:
        rows = f.readlines()

    if composite == False:
        for i in range(0, len(rows)):
            if rows[i][15:21] == 'CENTER' and int(rows[i].split()[0]) in elem_id:
                row_z1 = rows[i].split()
                row_z2 = rows[i+1].split()
                new_row = np.empty((1, 19))
                new_row[0, [3, 5, 6, 9, 11, 12, 15, 17, 18]] = 0
                new_row[0, 0] = elem_dict[int(row_z1[0])]
                new_row[0, 1:3] = [
                    (float(row_z1[_]) + float(row_z2[_-2]))/2 for _ in range(3, 5)]
                new_row[0, 4] = (float(row_z1[5]) + float(row_z2[5-2]))/2
                new_row[0, 13:15] = [float(row_z1[_]) for _ in range(3, 5)]
                new_row[0, 16] = float(row_z1[5])
                new_row[0, 7:9] = [float(row_z2[_-2]) for _ in range(3, 5)]
                new_row[0, 10] = float(row_z2[5-2])
                result_data = np.vstack((result_data, new_row))
    elif ply_num == 2:
        for i in range(0, len(rows)):
            if len(rows[i].split()) > 1 and rows[i].split()[1] in ply_id and rows[i].split()[1] == rows[i][16:17] and int(rows[i].split()[0]) in elem_id:
                row_z1 = rows[i].split()
                row_z2 = rows[i+1].split()
                new_row = np.empty((1, 19))
                new_row[0, [3, 9, 15]] = 0
                new_row[0, 0] = elem_dict[int(row_z1[0])]
                new_row[0, 1:3] = [
                    (float(row_z1[_]) + float(row_z2[_-1]))/2 for _ in range(2, 4)]
                new_row[0, 4] = (float(row_z1[4]) + float(row_z2[4-1]))/2
                new_row[0, 5:7] = [
                    (float(row_z1[_]) + float(row_z2[_-1]))/2 for _ in range(6, 4, -1)]
                new_row[0, 7:9] = [float(row_z1[_]) for _ in range(2, 4)]
                new_row[0, 10] = float(row_z1[4])
                new_row[0, 11:13] = [float(row_z1[_]) for _ in range(6, 4, -1)]
                new_row[0, 13:15] = [float(row_z2[_-1]) for _ in range(2, 4)]
                new_row[0, 16] = float(row_z2[4-1])
                new_row[0, 17:19] = [float(row_z2[_-1])
                                     for _ in range(6, 4, -1)]
                result_data = np.vstack((result_data, new_row))
    elif ply_num == 3:
        for i in range(0, len(rows)):
            if len(rows[i].split()) > 1 and rows[i].split()[1] in ply_id and rows[i].split()[1] == rows[i][16:17] and int(rows[i].split()[0]) in elem_id:
                row_z1 = rows[i].split()
                row_z2 = rows[i+1].split()
                row_z3 = rows[i+2].split()
                if len(row_z2) == 0:
                    row_z2 = rows[i+19].split()
                    row_z3 = rows[i+20].split()
                elif len(row_z3) == 0:
                    row_z3 = rows[i+20].split()
                new_row = np.empty((1, 19))
                new_row[0, [3, 5, 6, 9, 11, 12, 15, 17, 18]] = 0
                new_row[0, 0] = elem_dict[int(row_z1[0])]
                new_row[0, 1:3] = [float(row_z2[_-1]) for _ in range(2, 4)]
                new_row[0, 4] = float(row_z2[4-1])
                new_row[0, 5:7] = [float(row_z2[_-1]) for _ in range(6, 4, -1)]
                new_row[0, 7:9] = [float(row_z1[_]) for _ in range(2, 4)]
                new_row[0, 10] = float(row_z1[4])
                new_row[0, 11:13] = [float(row_z1[_]) for _ in range(6, 4, -1)]
                new_row[0, 13:15] = [float(row_z3[_-1]) for _ in range(2, 4)]
                new_row[0, 16] = float(row_z3[4-1])
                new_row[0, 17:19] = [float(row_z3[_-1])
                                     for _ in range(6, 4, -1)]
                result_data = np.vstack((result_data, new_row))

    result_data = np.delete(result_data, 0, axis=0)
    result_data = result_data[np.argsort(result_data[:, 0])]

    # Perform result coordinate conversion according to ANSYS definition
    degs = np.zeros((len(elem_data), 3))
    if coords_sys == 'local':
        for i in range(0, len(elem_data)):
            if np.unique(elem_data[i, 1:5]).shape[0] == 4:
                A = node_data[(node_data[:, 0] == elem_data[i, 3]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4]
                B = node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4]
                degs[i, 0] = np.rad2deg(
                    np.arccos(np.inner(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))))
                A = node_data[(node_data[:, 0] == elem_data[i, 4]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4]
                B = node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4]
                degs[i, 1] = np.rad2deg(
                    np.arccos(np.inner(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))))
                degs[i, 2] = (-degs[i, 0] + degs[i, 1]) / 2
            else:
                degs[i, :] = 0
    elif coords_sys == 'global':
        for i in range(0, len(elem_data)):
            A = node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4] - \
                node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4]
            B = np.array([1, 0, 0])
            degs[i, 2] = np.rad2deg(
                np.arccos(np.inner(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))))

    theta = np.deg2rad(degs[:, 2])
    result_data_ct = result_data.copy()
    result_data_ct[:, 1] = result_data[:, 1]*np.cos(theta)**2 + result_data[:, 2]*np.sin(
        theta)**2 + result_data[:, 4]*np.sin(2*theta)/k
    result_data_ct[:, 2] = result_data[:, 1]*np.sin(theta)**2 + result_data[:, 2]*np.cos(
        theta)**2 - result_data[:, 4]*np.sin(2*theta)/k
    result_data_ct[:, 4] = -(result_data[:, 1] - result_data[:, 2]) * \
        np.sin(2*theta) + result_data[:, 4]*np.cos(2*theta)
    result_data_ct[:, 7] = result_data[:, 7]*np.cos(theta)**2 + result_data[:, 8]*np.sin(
        theta)**2 + result_data[:, 10]*np.sin(2*theta)/k
    result_data_ct[:, 8] = result_data[:, 7]*np.sin(theta)**2 + result_data[:, 8]*np.cos(
        theta)**2 - result_data[:, 10]*np.sin(2*theta)/k
    result_data_ct[:, 10] = -(result_data[:, 7] - result_data[:, 8]) * \
        np.sin(2*theta) + result_data[:, 10]*np.cos(2*theta)
    result_data_ct[:, 13] = result_data[:, 13]*np.cos(
        theta)**2 + result_data[:, 14]*np.sin(theta)**2 + result_data[:, 16]*np.sin(2*theta)/k
    result_data_ct[:, 14] = result_data[:, 13]*np.sin(
        theta)**2 + result_data[:, 14]*np.cos(theta)**2 - result_data[:, 16]*np.sin(2*theta)/k
    result_data_ct[:, 16] = -(result_data[:, 13] - result_data[:, 14]) * \
        np.sin(2*theta) + result_data[:, 16]*np.cos(2*theta)

    disp_data = np.empty((1, 7))
    with open(output_path, 'r') as f:
        for row in f:
            if (row[20:23] == ' 0 ' or row[19:22] == ' G ') and int(row.split()[0]) in node_id:
                row_ = row.split()
                new_row = np.empty((1, 7))
                new_row[0, 0] = node_dict[int(row_[0])]
                new_row[0, 1:7] = [float(_) for _ in row_[2:8]]
                disp_data = np.vstack((disp_data, new_row))
    disp_data = np.delete(disp_data, 0, axis=0)

    return result_data_ct, disp_data


def write_files(input_dir, node_data=[], elem_data=[], strain_data=[], disp_data=[]):
    """Write node, element, strain, displacement data to csv files. These files are iFEM input file.

    Args:
        input_dir (str): input file directory of iFEM analysis.
        node_data (array[float], optional): node_data of iFEM input format.
        elem_data (array[float], optional): elem_data of iFEM input format.
        strain_data (array[float], optional): strain_data of iFEM input format.
        disp_data (array[float], optional): node_id and node displacement of 6 DOFs.
    """
    if len(node_data) != 0:
        ids = np.ones((len(node_data), 6)) * 1.27*10**30
        node_data_ = np.hstack((node_data, ids))
        pd.DataFrame(node_data_).to_csv(input_dir+'node_data.csv', header=[
            'node_id', 'x', 'y', 'z', 'id_ux', 'id_uy', 'id_uz', 'id_rx', 'id_ry', 'id_rz'], index=False)
    if len(elem_data) != 0:
        pd.DataFrame(elem_data).to_csv(input_dir+'elem_data.csv', na_rep='nan', header=[
            'elem_id', 'connection1', 'connection2', 'connection3', 'connection4', 'thickness', 'r3_in_plane'], index=False)
    if len(strain_data) != 0:
        pd.DataFrame(strain_data).to_csv(input_dir+'strain_data.csv', header=['elem_id', 'strain_mid_x', 'strain_mid_y', 'strain_mid_z', 'strain_mid_xy', 'strain_mid_yz', 'strain_mid_xz',
                                                                              'strain_top_x', 'strain_top_y', 'strain_top_z', 'strain_top_xy', 'strain_top_yz', 'strain_top_xz',
                                                                              'strain_bot_x', 'strain_bot_y', 'strain_bot_z', 'strain_bot_xy', 'strain_bot_yz', 'strain_bot_xz'], index=False)
    if len(disp_data) != 0:
        if disp_data.shape[1] == 7:
            pd.DataFrame(disp_data).to_csv(input_dir+'disp_data.csv', header=[
                'node_id', 'disp_x', 'disp_y', 'disp_z', 'rot_x', 'rot_y', 'rot_z'], index=False)
        elif disp_data.shape[1] == 4:
            pd.DataFrame(disp_data).to_csv(input_dir+'disp_data.csv',
                                           header=['node_id', 'disp_x', 'disp_y', 'disp_z'], index=False)

    return None


def import_stress(output_path):
    """Stress data from Nastran output file(.OUT). Stress must be output in the output file. (Both STRESS and STRAIN cannot be requested in the same subcase.)

    Args:
        output_path (str): the path of Nastran output file (.OUT)

    Returns:
        array: elem_id and stress of x, y, xy and von mises.
    """
    stress_data = np.empty((1, 13))
    with open(output_path, 'r') as f:
        rows = f.readlines()
    for i in range(0, len(rows)):
        if rows[i][15:21] == 'CENTER' and int(rows[i].split()[0]) in elem_id:
            row_z1 = rows[i].split()
            row_z2 = rows[i+1].split()
            new_row = np.empty((1, 13))
            new_row[0, 0] = elem_dict[int(row_z1[0])]
            new_row[0, 1] = (float(row_z1[3]) + float(row_z2[3-2]))/2
            new_row[0, 2] = (float(row_z1[4]) + float(row_z2[4-2]))/2
            new_row[0, 3] = (float(row_z1[5]) + float(row_z2[5-2]))/2
            new_row[0, 4] = (float(row_z1[9]) + float(row_z2[9-2]))/2
            new_row[0, 5] = float(row_z2[3-2])
            new_row[0, 6] = float(row_z2[4-2])
            new_row[0, 7] = float(row_z2[5-2])
            new_row[0, 8] = float(row_z2[9-2])
            new_row[0, 9] = float(row_z1[3])
            new_row[0, 10] = float(row_z1[4])
            new_row[0, 11] = float(row_z1[5])
            new_row[0, 12] = float(row_z1[9])
            stress_data = np.vstack((stress_data, new_row))
    stress_data = np.delete(stress_data, 0, axis=0)
    stress_data = stress_data[np.argsort(stress_data[:, 0])]

    return stress_data


def calc_elem_center(node, elem):
    """Calculate the coordinates of element centers from node_data and elem_data.

    Args:
        node (array[float]): node_data
        elem (array[float]): elem_data

    Returns:
        array: elem_id and the centroind coordinates of elements
    """
    elem_nodes = elem[:, 1:5].astype('int')
    elem_definition = [np.unique(_elem) for _elem in elem_nodes]
    elem_center = np.hstack([
        np.arange(1, len(elem)+1).reshape((-1, 1)),
        np.array([np.mean(node[_elem_def-1, 1:], axis=0)
                 for _elem_def in elem_definition])
    ])
    return elem_center


def geom_check(node_data, elem_center):
    """Alias of function: geom_checker."""
    print('This function will be removed. Please use "geom_checker" instead.')
    geom_checker(node_data, elem_center)


def geom_checker(node_data, elem_center):
    """geometry check in visualization

    Args:
        node_data (array[float]): node coordinates
        elem_center (array[float]): coordinates of element center
    """
    plt.figure(figsize=(20, 2))
    plt.subplot(141)
    plt.scatter(node_data[:, 1], node_data[:, 3], marker='.', color='gray')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.axis('equal')
    plt.subplot(142)
    plt.scatter(node_data[:, 1], node_data[:, 2], marker='.', color='gray')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.axis('equal')
    plt.subplot(143)
    plt.scatter(elem_center[:, 1], elem_center[:, 3], marker='.', color='gray')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.axis('equal')
    plt.subplot(144)
    plt.scatter(elem_center[:, 1], elem_center[:, 2], marker='.', color='gray')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.axis('equal')
    plt.show()


def calc_bulk_data(node_data, elem_data):
    """format node and element data to nastran input file format

    Args:
        node_data (array[float]): node coordinates
        elem_data (array[float]): element data

    Returns:
        array: pshell, shell and grid data
    """
    thickness = np.unique(elem_data[:, 5])
    shell_list = {thickness[i]: 100+i for i in range(len(thickness))}

    pshell = [p('PSHELL') + p(_shell_id) + p(201) + p(_thick) + p(201) + '\n'
              for _thick, _shell_id in shell_list.items()]

    shell = []
    for i in range(0, len(elem_data)):
        if len(np.unique(elem_data[i, 1:5])) == 4:
            row = p('CQUAD4') + p(str(int(elem_data[i, 0]))) + p(str(shell_list[elem_data[i, 5]])) \
                + p(str(int(elem_data[i, 1]))) + p(str(int(elem_data[i, 2]))) \
                + p(str(int(elem_data[i, 3]))) + p(str(int(elem_data[i, 4]))) \
                + p(str(0.)) + p(str(0.)) + '\n'
        else:
            row = p('CTRIA3') + p(str(int(elem_data[i, 0]))) + p(str(str(shell_list[elem_data[i, 5]]))) \
                + p(str(int(elem_data[i, 1]))) + p(str(int(elem_data[i, 2]))) \
                + p(str(int(elem_data[i, 3]))) \
                + p(str(0.)) + p(str(0.)) + '\n'
        shell.append(row)

    grid = []
    for i in range(0, len(node_data)):
        row = p('GRID') + p(str(int(node_data[i, 0]))) + p(str(0)) \
            + p(str(round(node_data[i, 1], 5))) + p(str(round(node_data[i, 2], 5))) + p(
                str(round(node_data[i, 3], 5))) + '\n'
        grid.append(row)

    return pshell, shell, grid


def p(string='', limit=8):
    """format string

    Args:
        string (str, optional): arbitrary string, the dafault, ''.

    Returns:
        str: the length is equal to limit
    """
    if len(str(string)) < limit:
        return str(string) + ' '*(limit-len(str(string)))
    else:
        return str(string)[:limit-1] + ' '


def create_spc(spc_nodes, dof):
    """create spc data

    Args:
        spc_nodes (array[float]): node identification numbers of spc nodes
        dof (str): constraint degree of freedom (e.g. '123456')

    Returns:
        list[str]: spc data of nastran input file format
    """
    spc_data = []
    count = 0
    init_ = True
    while True:
        if count == len(spc_nodes):
            break
        if init_:
            init_ = False
            row = p('SPC1') + p(1000) + p(dof)
            for j in range(6):
                if count == len(spc_nodes):
                    break
                row += p(int(spc_nodes[count]))
                count += 1
        else:
            row = p('')
            for j in range(8):
                if count == len(spc_nodes):
                    break
                row += p(int(spc_nodes[count]))
                count += 1
        row += '\n'
        spc_data.append(row)
    return spc_data


def export_spcd(spc_nodes, disp):
    """Export SPCD data for Re-FEM analysis.

    Args:
        spc_nodes (array[float]): The array of node identification numbers of spc nodes.
        disp (array[float]): The array of node displacement, which shape is (n, 6)

    Returns:
        list[str]: The list of SPC and SPCD data.
    """
    if len(spc_nodes) != len(disp):
        raise ValueError(
            f'the length of disp and node is not matched; {len(spc_nodes)} != {len(disp)}')

    spc = ''.join(create_spc(spc_nodes, 123))
    for i in range(len(spc_nodes)):
        spc += p('SPCD') + p(2000) + p(int(spc_nodes[i])) + p(1) + p(
            disp[i, 0]) + p(int(spc_nodes[i])) + p(2) + p(disp[i, 1]) + '\n'
        spc += p('SPCD') + p(2000) + \
            p(int(spc_nodes[i])) + p(3) + p(disp[i, 2]) + '\n'
    return spc


def create_rbe(independent_node_id, dependent_node_ids, dof, elem_id=None):
    """create rigid link

    Args:
        independent_node_id (int): independent node id number
        dependent_node_ids (list or np.array): list of dependent node ids
        dof (str): degree of freedom to constraint
        elem_id (int): arbitrary element id.

    Returns:
        list[str]: rigid link data
    """
    _elem_id = 1 if elem_id is None else elem_id
    assert isinstance(_elem_id, int), 'elem_id must be integer.'
    i = 0
    rbe = []
    for row in range(0, int((4+len(dependent_node_ids))/8)+1):
        line = ''
        if row == 0:
            line = p('RBE2') + p(_elem_id) + \
                p(int(independent_node_id)) + p(dof)
            _ = 5 if len(dependent_node_ids) >= 5 else len(dependent_node_ids)
            for j in range(_):
                line += p(int(dependent_node_ids[i]))
                i += 1
        else:
            for j in range(9):
                if j == 0:
                    line += p('')
                else:
                    try:
                        line += p(int(dependent_node_ids[i]))
                        i += 1
                    except IndexError:
                        break
        rbe.append(line + '\n')
    return rbe


def create_pload(load_elems, pressure, load_id=2000):
    """Create pressure load cards.

    Args:
        load_elems (list): The list of element ids to apply forces.
        pressure (float or list): The magnitude of pressure load.
        load_id (int): Load set identification number, the default is 2000.

    Returns:
        list: The list which consist of PLAOD4 cards.
    """
    if type(pressure) in (int, float):
        pressure = [pressure] * len(load_elems)
    if len(load_elems) != len(pressure):
        raise ValueError(
            f'The length of load_elems and pressure are not agreed. {len(load_elems)} != {pressure}')

    pload = [
        p('PLOAD4') + p(int(load_id)) +
        p(int(load_elems[i])) + p(float(pressure[i])) + '\n'
        for i in range(len(load_elems))
    ]
    return pload


def create_nas_input(nas_file_path, node_data, elem_data, spc_data='', load_data='', mat_data=''):
    """Create Nastrain input file (.nas) from node_data and elem_data. This file is only used in paraview for visualize, so material data should be modified to use for analysis.

    Args:
        nas_file_path (str): the path of new file (.nas)
        node_data (array[float]):
        elem_data (array[float]):
        spc_data (list[str], optional): spc data (for stractural analysis)
        load_data (list[str], optional): load data (for stractural analysis)
    """
    pshell, shell, grid = calc_bulk_data(node_data, elem_data)

    initialization = ['SOL 101', 'CEND',
                      'ECHO = NONE', 'RESVEC = YES', 'SUBCASE 1']
    case_control = ['SUBTITLE=static', 'SPC = 1000', 'LOAD = 2000',
                    'DISPLACEMENT(PRINT,REAL)=ALL', 'STRAIN(PRINT,CENTER,FIBER,REAL)=ALL',
                    '$STRESS(PRINT,CENTER,FIBER,REAL)=ALL']
    bulk_data = ['BEGIN BULK',
                 'PARAM    POST    -1',
                 'PARAM,GRDPNT,-1',]

    if mat_data == '':
        mat_data = ['MAT1    201     210000.         0.3     7.85-9\n',]

    # create new file
    with open(nas_file_path, mode='w') as f:
        for _ in initialization:
            f.write(_+'\n')

        for _ in case_control:
            f.write('  '+_+'\n')
        f.write('\n')

        for _ in bulk_data:
            f.write(_+'\n')
        for data in (spc_data, load_data, pshell, mat_data, shell, grid):
            f.write('\n')
            f.writelines(data)
        f.write('ENDDATA')


def id_changer(part_node, part_elem):
    """Change node_id and elem_id to serial number.

    Args:
        part_node (array): node_data which have non serial node_id.
        part_elem (array): elem_data which have non serial elem_id and node_id in part_elem must be corresponding with node_id in part_node.

    Returns:
        array: node_data, elem_data
    """
    part_node = part_node.copy()
    part_elem = part_elem.copy()
    part_node_dict = dict(zip(part_node[:, 0], np.arange(1, len(part_node)+1)))
    part_elem[:, 0] = np.arange(1, len(part_elem)+1)
    part_node[:, 0] = np.arange(1, len(part_node)+1)
    for i in range(0, len(part_elem)):
        for j in range(1, 5):
            part_elem[i, j] = part_node_dict[part_elem[i, j]]
    return part_node, part_elem


def node_displacement(output, case_start, case_num, model, verbal=True):
    """Pick node_displacement data up from nastran output file which have some subcases.

    Args:
        output (str): The file path of .OUT file.
        case_start: The first subcase number of analysis.
        case_num: The number of subcases.
        model (instance): The structural model including node and element information.

    Returns:
        array: displacement data for all subcases. If case_num > 1, array has 3 dimentions.
    """
    node_data = model.node
    node_id = list(model.node_dict.keys())
    node_dict = model.node_dict

    with open(output, 'r') as f:
        output = f.readlines()

    for case in range(case_start, case_start+case_num):
        print_log('[disp] Case '+str(case)+' is starting...', verbal)
        disp_data = []
        trigger = False
        for row in output:
            # if 'SUBCASE '+str(case) in row:
            #    trigger = True
            # if 'SUBCASE '+str(case+1) in row or 'SUBCASE 9999999' in row:
            #    break
            if row[69:76] == 'SUBCASE':
                if int(row[77:84]) == case:
                    trigger = True
                if int(row[77:84]) != case and trigger:
                    break

            if trigger:
                if row[20:23] == ' 0 ' and int(row.split()[0]) in node_id:
                    row_ = row.split()
                    new_row = []
                    new_row += [node_dict[int(row_[0])]]
                    new_row += [float(_) for _ in row_[2:8]]
                    disp_data.append(new_row)
        disp_data_ = np.array(disp_data)

        disp_data = np.zeros((len(node_data), 7))
        disp_data[:, 0] = node_data[:, 0].copy()
        for i in range(0, len(disp_data)):
            if disp_data[i, 0] in disp_data_[:, 0]:
                disp_data[i, 1:] = disp_data_[
                    disp_data_[:, 0] == disp_data[i, 0], 1:]

        if case == case_start:
            disp_data_all = disp_data
        else:
            disp_data_all = np.dstack((disp_data_all, disp_data))

    return disp_data_all


def elem_strain(output, case_start, case_num, model, verbal=True):
    """Pick pshell strain at each element data up from nastran output file which have some subcases.

    Args:
        output (str): The file path of .OUT file.
        case_start (int): The first subcase number of analysis.
        case_num (int): The number of subcases.
        model (instance): The structural model including node and element information.

    Returns:
        array: strain data for all subcases. If case_num > 1, array has 3 dimentions.
    """
    node_data = model.node
    elem_data = model.elem
    elem_id = list(model.elem_dict.keys())
    elem_dict = model.elem_dict
    pshell_bool = np.array(
        [model.part_dict[i]['type'] == 'PSHELL' for i in model.elem_parts])
    elem_data = elem_data[pshell_bool]
    elem_id = [elem_id[i] for i, _ in enumerate(pshell_bool) if _]
    elem_dict = dict(zip(
        [list(elem_dict.keys())[i] for i, _ in enumerate(pshell_bool) if _],
        [list(elem_dict.values())[i] for i, _ in enumerate(pshell_bool) if _]
    ))

    with open(output, 'r') as f:
        output = f.readlines()

    for case in range(case_start, case_start+case_num):
        print_log('[strain] Case '+str(case)+' is starting...', verbal)
        strain_data = np.empty((1, 19))

        trigger_1 = False
        trigger_2 = False
        for i, row in enumerate(output):
            if row.endswith('M A X I M U M   D I S P L A C E M E N T S\n'):
                trigger_1 = True

            if trigger_1:
                if row.endswith('SUBCASE '+str(case)+'\n'):
                    trigger_2 = True

            #    if row.endswith('SUBCASE '+str(case+1)+'\n'):
            #        break
            #    if (trigger_2) & (row.endswith('SUBCASE 9999999\n')):
            #        break
            if trigger_2:
                if row[69:76] == 'SUBCASE':
                    if int(row[77:84]) != case:
                        break

            if trigger_1 & trigger_2:
                if row[15:21] == 'CENTER' and int(row.split()[0]) in elem_id:
                    row_z1 = output[i].split()
                    row_z2 = output[i+1].split()
                    new_row = np.empty((1, 19))
                    new_row[0, [3, 5, 6, 9, 11, 12, 15, 17, 18]] = 0
                    new_row[0, 0] = elem_dict[int(row_z1[0])]
                    new_row[0, 1:3] = [
                        (float(row_z1[_]) + float(row_z2[_-2]))/2 for _ in range(3, 5)]
                    new_row[0, 4] = (float(row_z1[5]) + float(row_z2[5-2]))/2
                    new_row[0, 13:15] = [float(row_z1[_]) for _ in range(3, 5)]
                    new_row[0, 16] = float(row_z1[5])
                    new_row[0, 7:9] = [float(row_z2[_-2]) for _ in range(3, 5)]
                    new_row[0, 10] = float(row_z2[5-2])
                    strain_data = np.vstack((strain_data, new_row))

            if len(strain_data) == len(elem_id)+1:
                break

        strain_data = np.delete(strain_data, 0, axis=0)
        strain_data = strain_data[np.argsort(strain_data[:, 0])]

        # Perform strain coordinate conversion according to ANSYS definition
        degs = np.empty((len(elem_data), 3))
        for i in range(0, len(elem_data)):
            if np.unique(elem_data[i, 1:5]).shape[0] == 4:
                A = node_data[(node_data[:, 0] == elem_data[i, 3]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4]
                B = node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4]
                degs[i, 0] = np.rad2deg(
                    np.arccos(np.inner(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))))
                A = node_data[(node_data[:, 0] == elem_data[i, 4]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4]
                B = node_data[(node_data[:, 0] == elem_data[i, 1]), 1:4] - \
                    node_data[(node_data[:, 0] == elem_data[i, 2]), 1:4]
                degs[i, 1] = np.rad2deg(
                    np.arccos(np.inner(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))))
                degs[i, 2] = (-degs[i, 0] + degs[i, 1]) / 2
            else:
                degs[i, :] = 0
        theta = np.deg2rad(degs[:, 2])
        strain_data_ct = strain_data.copy()
        strain_data_ct[:, 1] = strain_data[:, 1]*np.cos(theta)**2 + strain_data[:, 2]*np.sin(
            theta)**2 + strain_data[:, 4]*np.sin(2*theta)/2
        strain_data_ct[:, 2] = strain_data[:, 1]*np.sin(theta)**2 + strain_data[:, 2]*np.cos(
            theta)**2 - strain_data[:, 4]*np.sin(2*theta)/2
        strain_data_ct[:, 4] = -(strain_data[:, 1] - strain_data[:, 2]) * \
            np.sin(2*theta) + strain_data[:, 4]*np.cos(2*theta)
        strain_data_ct[:, 7] = strain_data[:, 7]*np.cos(theta)**2 + strain_data[:, 8]*np.sin(
            theta)**2 + strain_data[:, 10]*np.sin(2*theta)/2
        strain_data_ct[:, 8] = strain_data[:, 7]*np.sin(theta)**2 + strain_data[:, 8]*np.cos(
            theta)**2 - strain_data[:, 10]*np.sin(2*theta)/2
        strain_data_ct[:, 10] = -(strain_data[:, 7] - strain_data[:, 8]) * \
            np.sin(2*theta) + strain_data[:, 10]*np.cos(2*theta)
        strain_data_ct[:, 13] = strain_data[:, 13]*np.cos(
            theta)**2 + strain_data[:, 14]*np.sin(theta)**2 + strain_data[:, 16]*np.sin(2*theta)/2
        strain_data_ct[:, 14] = strain_data[:, 13]*np.sin(
            theta)**2 + strain_data[:, 14]*np.cos(theta)**2 - strain_data[:, 16]*np.sin(2*theta)/2
        strain_data_ct[:, 16] = -(strain_data[:, 13] - strain_data[:, 14]) * \
            np.sin(2*theta) + strain_data[:, 16]*np.cos(2*theta)

        if case == case_start:
            strain_data_all = strain_data_ct
        else:
            strain_data_all = np.dstack((strain_data_all, strain_data_ct))

    return strain_data_all


def elem_strain_pcomp(output_file, case_start, case_num, model, verbal=True):
    """Pick strain at each element data up from nastran output file which have some subcases.

    Args:
        output (str): File path of .OUT file.
        case_start: The first subcase number of analysis.
        case_num: The number of subcases.
        model (instance): The structural model including node and element information.

    Returns:
        array: strain data for all subcases. If case_num > 1, array has 3 dimentions.
    """
    elem_dict = model.elem_dict

    with open(output_file, 'r') as f:
        output = f.readlines()

    for case in range(case_start, case_start+case_num):
        print_log('[strain] Case '+str(case)+' is starting...', verbal)
        strain = list()
        trigger_1 = False
        trigger_2 = False
        trigger_3 = False
        for i, row in enumerate(output):
            # Enter strain output part.
            if row.endswith('M A X I M U M   D I S P L A C E M E N T S\n'):
                trigger_1 = True
            # Enter target case output part.
            if trigger_1:
                if row.endswith('SUBCASE '+str(case)+'\n'):
                    trigger_2 = True
            # Exit target case output part.
            if trigger_2:
                if row[69:76] == 'SUBCASE':
                    if int(row[77:84]) != case:
                        break
            # Enter each strain value part.
            if trigger_1 & trigger_2:
                if len(row.split()) == 11:
                    try:
                        if int(row.split()[0]) in elem_dict.keys():
                            trigger_3 = True
                    except:
                        pass
            # Pass blank row.
            if trigger_3:
                if row == ' \n':
                    trigger_3 = False
            # Append strain data.
            if trigger_3:
                strain.append(row)
            # After blank row and headers.
            if trigger_1 & trigger_2:
                if len(row.split()) in (12, 14):
                    if row.split()[0] == 'ID':
                        trigger_3 = True

        # Convert format.
        strains = list()
        for row in strain:
            items = [float(item) for item in row.split()]
            if len(items) == 11:
                _elem_id = elem_dict[int(items[0])]
                items[0] = _elem_id
            elif len(items) == 10:
                items = [_elem_id] + items
            else:
                raise ValueError(row)
            strains.append(items)
        strains = np.array(strains)
        df = pd.DataFrame(strains)

        def extract_strain(df):
            mid = 2 if len(df) == 3 else 3 if len(df) == 5 else 0  # ply_id
            if mid == 0:
                print(df)
                raise ValueError(f'Unknown plies. => {len(df)}')
            row = sum([
                [df[2].iloc[j], df[3].iloc[j], 0., df[4].iloc[j], 0., 0.]
                for j in (mid-1, 0, -1)
            ], [])
            return row

        strains = df.iloc[:, 0:5].groupby(0).apply(extract_strain)
        strains = np.hstack([np.array(strains.index).reshape(-1, 1),
                            np.vstack([row for row in strains])])  # (n*19)

        if case == case_start:
            if case_num == 1:
                strain_all = strains
            else:
                strain_all = strains.reshape(-1, 19, 1)
        else:
            strain_all = np.dstack([strain_all, strains.reshape(-1, 19, 1)])
    print_log('[strain] Successfully finished.', verbal)
    return strain_all


def _float_conv(num):
    try:
        num_ = float(num)
    except:
        if '-' in num:
            ele = num.split('-')
            if ele[0] == '':
                num_ = float('-'+ele[1]+'e-'+ele[2])
            else:
                num_ = float(ele[0]+'e-'+ele[1])
        elif '+' in num:
            ele = num.split('+')
            if ele[0] == '':
                num_ = float('-'+ele[1]+'e+'+ele[2])
            else:
                num_ = float(ele[0]+'e+'+ele[1])
        elif num == '' or len(str(num).split()) == 0:  # '' or ' '*n
            num_ = num
        else:
            raise ValueError(
                f'Cannot convert input value to float number: "{num}"')
    return num_


def model_builder(input_path, return_all=False):
    """Import node & elem data from Nastran model data file (.BDF).

    Args:
        input_path (str): the file path of Nastran model data file (.BDF).
        return_all (bool, optional): If True, dictionaries are returned together with node_data and elem_data.

    Returns:
        array: node_data, elem_data, node_dict, node_idict, elem_dict, elem_idict
    """
    print_log('This function is deprecated. Please use "import_model" instead.')
    try:
        with open(input_path) as f:
            model_data = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, encoding='Shift-JIS') as f:
            model_data = f.readlines()

    shell_id = []
    shell_thickness = []
    for _ in range(0, len(model_data)):
        if model_data[_][:6] == 'PSHELL':
            row = model_data[_].split()
            shell_id.append(row[1])
            shell_thickness.append(row[3])
        elif model_data[_][:5] == 'PCOMP':
            row = model_data[_].split()
            shell_id.append(row[1])
            shell_thickness.append(str(float(row[2])*2))
    shell_dict = dict(zip(shell_id, shell_thickness))

    global node_data, elem_data
    node_data = []
    elem_data = []

    for _ in range(0, len(model_data)):
        if model_data[_][:5] == 'GRID ':
            row = model_data[_]
            node_data.append([int(row[8:16]), _float_conv(
                row[24:32]), _float_conv(row[32:40]), _float_conv(row[40:48])])
        elif model_data[_][:5] == 'GRID*':
            row1 = model_data[_].split()
            row2 = model_data[_+1].split()
            try:
                node_data.append([int(row1[1]), _float_conv(
                    row1[2]), _float_conv(row1[3]), _float_conv(row2[1])])
            except:
                if len(row1[2]) == 32:
                    node_data.append([int(row1[1]), _float_conv(
                        row1[2][:16]), _float_conv(row1[2][16:]), _float_conv(row2[1])])
                else:
                    node_data.append([int(row1[1]), _float_conv(
                        row1[2][:15]), _float_conv(row1[2][15:]), _float_conv(row2[1])])

        if model_data[_][:7] == 'CQUAD4*':
            row1 = model_data[_].split()
            row2 = model_data[_+1].split()
            elem_data.append([int(row1[1]), int(row1[3]), int(row1[4]), int(
                row2[1]), int(row2[2]), float(shell_dict[row1[2]]), int(1)])

        elif model_data[_][:7] == 'CQUAD4 ':
            row = model_data[_].split()
            elem_data.append([int(row[1]), int(row[3]), int(row[4]), int(
                row[5]), int(row[6]), float(shell_dict[row[2]]), int(1)])

        elif model_data[_][:7] == 'CTRIA3 ':
            row = model_data[_].split()
            elem_data.append([int(row[1]), int(row[3]), int(row[4]), int(
                row[5]), int(row[5]), float(shell_dict[row[2]]), int(1)])

    node_data = np.array(node_data)
    node_data = node_data[np.argsort(node_data[:, 0])]
    elem_data = np.array(elem_data)
    elem_data = elem_data[np.argsort(elem_data[:, 0])]

    _ = elem_data[:, 1:5]
    remove = [True if node_data[i, 0]
              in _ else False for i in range(0, len(node_data))]
    node_data = node_data[remove]

    global node_id, node_dict, node_idict, elem_id, elem_dict, elem_idict

    node_id = node_data[:, 0].copy()
    node_data[:, 0] = np.arange(1, len(node_data)+1)
    node_dict = dict(zip(node_id, node_data[:, 0]))
    node_idict = dict(zip(node_data[:, 0], node_id))

    elem_id = elem_data[:, 0].copy()
    elem_data[:, 0] = np.arange(1, len(elem_data)+1)
    elem_dict = dict(zip(elem_id, elem_data[:, 0]))
    elem_idict = dict(zip(elem_data[:, 0], elem_id))

    for i in range(0, len(elem_data)):
        for j in range(1, 5):
            elem_data[i, j] = node_dict[int(elem_data[i, j])]

    if return_all:
        return node_data, elem_data, node_dict, node_idict, elem_dict, elem_idict
    else:
        return node_data, elem_data


def calc_degs(vector, node_data, elem_data, direction):
    """Calculate degrees between vectors and x, y or z axis.

    Args:
        vector (str): 'normal' or 'basis'
        node_data (array[float]): node_id(serial number) and coordinates. iFEM input file format.
        elem_data (array[float]): elem_id(serial number), elememnt dinifition(vertex node_id), thickness and r3 in plane. iFEM input file format.
        direction (str): 'x', 'y' or 'z'.

    Returns:
        array: 2d vector contains degree between vectors and x, y or z axis.
    """
    if direction == 'x':
        V = np.array([1, 0, 0])
    elif direction == 'y':
        V = np.array([0, 1, 0])
    elif direction == 'z':
        V = np.array([0, 0, 1])

    degs = np.empty((len(elem_data)))
    if vector == 'normal':
        for i in range(0, len(elem_data)):
            A = node_data[int(elem_data[i, 2])-1, 1:4] - \
                node_data[int(elem_data[i, 1])-1, 1:4]
            B = node_data[int(elem_data[i, 4])-1, 1:4] - \
                node_data[int(elem_data[i, 1])-1, 1:4]
            C = np.cross(A, B)
            deg = np.rad2deg(
                np.arccos(C @ V / (np.linalg.norm(C) * np.linalg.norm(V))))
            degs[i] = deg
    elif vector == 'basis':
        for i in range(0, len(elem_data)):
            A = node_data[int(elem_data[i, 2])-1, 1:4] - \
                node_data[int(elem_data[i, 1])-1, 1:4]
            deg = np.rad2deg(
                np.arccos(A @ V / (np.linalg.norm(A) * np.linalg.norm(V))))
            degs[i] = deg
    return degs


def R_matrix(deg, rank):
    """Create rotation matrix.

    Args:
        deg (array): degree of rotation(counterclockwise is a plus number)
        rank (int): rank of rotation matrix

    Returns:
        array: rotation matrix
    """
    rad = np.deg2rad(deg)
    if rank == 2:
        R = np.array([[np.cos(rad), -np.sin(rad)],
                      [np.sin(rad), np.cos(rad)]])
    elif rank == 3:
        pass
        # miscode / To Do
        # R_x = np.array([[1, 0, 0],
        #                 [0, np.cos(rad), -np.sin(rad)],
        #                 [0, np.sin(rad), np.cos(rad)]])
        # R_y = np.array([[np.cos(rad), 0, np.sin(rad)],
        #                 [0, 1, 0],
        #                 [-np.sin(rad), 0, np.cos(rad)]])
        # R_z = np.array([[np.cos(rad), -np.sin(rad), 0],
        #                 [np.sin(rad), np.cos(rad), 0],
        #                 [0, 0, 1]])
        # R = R_x @ R_y @ R_z

    return R


def calc_basis_iqs4(fe_node, _connection):
    """calculate coordinate basis

    Args:
        fe_node (pd.DataFrame): node data of structural model
        _connection (list): connection of one element

    Returns:
        list: system of quadshell element
    """
    node1 = fe_node.loc[_connection[0], :].values
    node2 = fe_node.loc[_connection[1], :].values
    node3 = fe_node.loc[_connection[2], :].values
    node4 = fe_node.loc[_connection[3], :].values
    nodes = np.array([node1, node2, node3, node4])

    # Compute the normal vector of the least-squares plane
    centroid = np.mean(nodes, axis=0)
    centroid_to_node = nodes - centroid
    w, v = np.linalg.eig(centroid_to_node.T@centroid_to_node)
    n_vector = v.T[np.argmin(w)]

    # Projecting the node vectors to the least-squares plane
    distance = centroid_to_node.dot(n_vector)
    distance = distance[:, np.newaxis]
    nodes = nodes - distance * n_vector.repeat(4).reshape(4, -1)

    # Find the intersection of the diagonals of a projected rectangle
    triangle124_area = np.linalg.norm(
        np.cross(nodes[3] - nodes[1], nodes[0] - nodes[1])) / 2
    triangle234_area = np.linalg.norm(
        np.cross(nodes[3] - nodes[1], nodes[2] - nodes[1])) / 2
    diagonal_prop = triangle124_area / (triangle124_area + triangle234_area)
    intersection_of_diagonal = nodes[0] + diagonal_prop * (nodes[2] - nodes[0])

    node2_vec = nodes[1] - intersection_of_diagonal
    node3_vec = nodes[2] - intersection_of_diagonal
    node2_vec_len = np.linalg.norm(node2_vec)
    node3_vec_len = np.linalg.norm(node3_vec)
    x_vector = (node2_vec_len * (node3_vec) + node3_vec_len * (node2_vec)) / \
        (node3_vec_len + node2_vec_len)
    x_vector /= np.linalg.norm(x_vector)
    z_vector = np.cross(x_vector, nodes[2] - nodes[0])
    z_vector /= np.linalg.norm(z_vector)
    y_vector = np.cross(z_vector, x_vector)
    return x_vector, y_vector, z_vector


def calc_basis_t1(fe_node, _connection):
    """calculate coordinate basis

    Args:
        fe_node (pd.DataFrame): node data of structural model
        _connection (list): connection of one element

    Returns:
        list: system of element
    """
    node1 = fe_node.loc[_connection[0], :].values
    node2 = fe_node.loc[_connection[1], :].values
    node3 = fe_node.loc[_connection[2], :].values

    x_vector = node2 - node1
    x_vector = x_vector / np.linalg.norm(x_vector)
    z_vector = np.cross(x_vector, node3 - node1)
    z_vector /= np.linalg.norm(z_vector)
    y_vector = np.cross(z_vector, x_vector)
    return x_vector, y_vector, z_vector


def _calc_elem_local_basis(node, _connection):
    """calculate elemental local coordinate basis.

    Args:
        node (array): node data of structural model, the shape is (n, 4).
        _connection (list): connection of one element.

    Returns:
        list: system of element.
    """
    calc_shell_basis = {
        3: calc_basis_t1,
        4: calc_basis_iqs4,
    }
    fe_node = pd.DataFrame(node).set_index(0)
    _connection = list(_connection)
    _connection = sorted(set(_connection), key=_connection.index)
    x_vector, y_vector, z_vector = calc_shell_basis[len(
        _connection)](fe_node, _connection)
    basis = np.vstack([x_vector, y_vector, z_vector]).T
    return basis


def calc_elem_local_basis(node, elem):
    """calculate elemental local coordinate basis.

    Args:
        node (array): node data of structural model in iFEM format.
        elem (array): elem data of structural model in iFEM format.

    Returns:
        array: The array of elemental local coordinate basis.
    """
    bases_elem_local = np.dstack([
        _calc_elem_local_basis(node, _connection)
        for _connection in elem[:, 1:5]
    ])
    return bases_elem_local


def _calc_elem_global_basis(node, _connection, global_axis):
    """calculate coordinate basis

    Args:
        node (pd.array): node data of structural model, the shape is (n, 4).
        _connection (list): connection of one element.
        global_axis (str): global coord system axis.

    Returns:
        list: system of one element
    """
    global_vec = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    global_vec = np.array(global_vec[global_axis])
    elem_z_vec = _calc_elem_local_basis(node, _connection)[:, 2]
    if abs(np.dot(global_vec.reshape((-1,)), elem_z_vec.reshape((-1,)))) == 1:
        # Do not rotate if global_vec is matched elem_z_vec.
        basis = np.vstack(np.eye(3))
    else:
        elem_vec0 = global_vec - \
            np.dot(global_vec.reshape((-1,)),
                   elem_z_vec.reshape((-1,)))*elem_z_vec
        elem_vec0 /= np.linalg.norm(elem_vec0)
        elem_vec1 = np.cross(elem_z_vec, elem_vec0)
        basis = np.vstack([elem_vec0, elem_vec1, elem_z_vec])
    return basis


def calc_elem_global_basis(node, elem):
    """calculate elemental local coordinate basis.

    Args:
        node (array): node data of structural model in iFEM format.
        elem (array): elem data of structural model in iFEM format.

    Returns:
        array: The array of elemental global coordinate basis.
    """
    bases_elem_global = np.dstack([
        _calc_elem_global_basis(node, _connection, 'x')
        for _connection in elem[:, 1:5]
    ])
    return bases_elem_global


def array2tensor(data, state_q, rank):
    """Create tensor from array

    Args:
        data (array): strain or stress data
        state_q (str): 'strain' or 'stress'
        rank (int): the rank of tensor, this value must be 2 or 3.

    Returns:
        array: tensor.
    """
    assert data.shape[
        1] == 6, f'the number of columns of data must be 6, not {data.shape[1]}.'
    k = 0.5 if state_q == 'strain' else 1

    def rs(vec):
        return vec.reshape((1, 1, -1))

    if rank == 2:
        tensor = np.vstack([
            np.hstack([rs(data[:, 0]), k*rs(data[:, 3])]),
            np.hstack([k*rs(data[:, 3]), rs(data[:, 1])]),
        ])
    elif rank == 3:
        tensor = np.vstack([
            np.hstack([rs(data[:, 0]), k*rs(data[:, 3]), k*rs(data[:, 5])]),
            np.hstack([k*rs(data[:, 3]), rs(data[:, 1]), k*rs(data[:, 4])]),
            np.hstack([k*rs(data[:, 5]), k*rs(data[:, 4]), rs(data[:, 2])]),
        ])
    else:
        raise ValueError('The parameter rank must be 2 or 3.')
    return tensor


def tensor2array(tensor, state_q, rank):
    """Create array from tensor

    Args:
        array: tensor
        state_q: 'strain' or 'stress'
        rank (int): the rank of tensor, this value must be 2 or 3.

    Returns:
        array: array that its shape is (n rows x 6 cols).
    """
    k = 2 if state_q == 'strain' else 1

    array = np.zeros((tensor.shape[2], 6))

    array[:, 0] = tensor[0, 0, :]
    array[:, 1] = tensor[1, 1, :]
    array[:, 3] = k * tensor[0, 1, :]
    if rank == 3:
        array[:, 2] = tensor[2, 2, :]
        array[:, 4] = k * tensor[1, 2, :]
        array[:, 5] = k * tensor[0, 2, :]
    return array


def rotate_strain(strain_before, bases_before, bases_after):
    """Rotate strain basis.

    Args:
        strain_before (array): The array of strain to rotate, the shape is (n, {6, 7, 19}, *).
        bases_before (array): The array of basis beform rotation, the shape is (3, 3, n).
        bases_after (array): The array of basis after rotation, the shape is (3, 3, n).

    Returns:
        array: The array of rotated strain.

    Example:
        >>> # Rotate strain coordinate system from elem_local to elem_global
        >>> node, elem = import_input(model_file_path) or model_builder(model_file_path)
        >>> bases_elem_local = calc_elem_local_basis(node, elem)
        >>> bases_elem_global = calc_elem_global_basis(node, elem)
        >>> rotated_strain = rotate_strain(strain, bases_elem_local, bases_elem_global)
    """
    if len(strain_before.shape) == 2:
        return _rotate_strain_phase(strain_before, bases_before, bases_after)
    elif len(strain_before.shape) == 3:
        strain_after = np.dstack([
            _rotate_strain_phase(strain_before[:, :, i], bases_before, bases_after)[
                :, :, np.newaxis]
            for i in range(strain_before.shape[2])
        ])
        return strain_after
    else:
        raise ValueError('The shape of strain is not supported.')


def _rotate_strain_phase(strain_before, bases_before, bases_after):
    """Rotate strain basis.

    Args:
        strain_before (array): The array of strain to rotate, the shape is (n, {6, 7, 19}).
        bases_before (array): The array of basis beform rotation, the shape is (3, 3, n).
        bases_after (array): The array of basis after rotation, the shape is (3, 3, n).

    Returns:
        array: The array of rotated strain.
    """
    cols = strain_before.shape[1]
    assert cols in (
        6, 7, 19), f'The number of columns of strain data must be (6, 7, 19), not {cols}.'
    r_matrix = np.einsum(
        'ijk,jlk->ilk', bases_before.transpose(1, 0, 2), bases_after)

    if cols == 6:
        strain_after = _rotate_strain(strain_before, r_matrix)
    elif cols == 7:
        strain_after = np.hstack([
            strain_before[:, 0:1],
            _rotate_strain(strain_before[:, 1:], r_matrix)
        ])
    elif cols == 19:
        strain_after = np.empty(strain_before.shape)
        strain_after[:, 0] = strain_before[:, 0]
        for i in range(3):
            strain_after[:, i*6+1:(i+1)*6+1] = \
                _rotate_strain(strain_before[:, i*6+1:(i+1)*6+1], r_matrix)
    else:
        raise ValueError('The shape of strain is not supported.')
    return strain_after


def _rotate_strain(strain_before, r_matrix):
    """Rotate strain basis using rotation matrix.

    Args:
        strain_before (array): The array of strain to rotate.
        r_matrix (array): Rotation matrix, the shape is (3, 3, n).

    Returns:
        array: The array of rotated strain.
    """
    tensor_before = array2tensor(strain_before, 'strain', rank=3)
    tensor_after = np.einsum(
        'ijk,jlk,lmk->imk', r_matrix.transpose(1, 0, 2), tensor_before, r_matrix)
    strain_after = tensor2array(tensor_after, 'strain', rank=3)
    return strain_after


def rotate_stress(stress_before, bases_before, bases_after):
    """Rotate stress basis.

    Args:
        stress_before (array): The array of stress to rotate, the shape is (n, {6, 7, 19}, *).
        bases_before (array): The array of basis beform rotation, the shape is (3, 3, n).
        bases_after (array): The array of basis after rotation, the shape is (3, 3, n).

    Returns:
        array: The array of rotated stress.

    Example:
        >>> # Rotate stress coordinate system from elem_local to elem_global
        >>> node, elem = import_input(model_file_path) or model_builder(model_file_path)
        >>> bases_elem_local = calc_elem_local_basis(node, elem)
        >>> bases_elem_global = calc_elem_global_basis(node, elem)
        >>> rotated_stress = rotate_stress(stress, bases_elem_local, bases_elem_global)
    """
    if len(stress_before.shape) == 2:
        return _rotate_stress_phase(stress_before, bases_before, bases_after)
    elif len(stress_before.shape) == 3:
        stress_after = np.dstack([
            _rotate_stress_phase(stress_before[:, :, i], bases_before, bases_after)[
                :, :, np.newaxis]
            for i in range(stress_before.shape[2])
        ])
        return stress_after
    else:
        raise ValueError('The shape of stress is not supported.')


def _rotate_stress_phase(stress_before, bases_before, bases_after):
    """Rotate stress basis.

    Args:
        stress_before (array): The array of stress to rotate, the shape is (n, {6, 7, 19}).
        bases_before (array): The array of basis beform rotation, the shape is (3, 3, n).
        bases_after (array): The array of basis after rotation, the shape is (3, 3, n).

    Returns:
        array: The array of rotated stress.
    """
    cols = stress_before.shape[1]
    assert cols in (
        6, 7, 19), f'The number of columns of stress data must be (6, 7, 19), not {cols}.'
    r_matrix = np.einsum(
        'ijk,jlk->ilk', bases_before.transpose(1, 0, 2), bases_after)

    if cols == 6:
        stress_after = _rotate_stress(stress_before, r_matrix)
    elif cols == 7:
        stress_after = np.hstack([
            stress_before[:, 0:1],
            _rotate_stress(stress_before[:, 1:], r_matrix)
        ])
    elif cols == 19:
        stress_after = np.empty(stress_before.shape)
        stress_after[:, 0] = stress_before[:, 0]
        for i in range(3):
            stress_after[:, i*6+1:(i+1)*6+1] = \
                _rotate_stress(stress_before[:, i*6+1:(i+1)*6+1], r_matrix)
    else:
        raise ValueError('The shape of stress is not supported.')
    return stress_after


def _rotate_stress(stress_before, r_matrix):
    """Rotate stress basis using rotation matrix.

    Args:
        stress_before (array): The array of stress to rotate.
        r_matrix (array): Rotation matrix, the shape is (3, 3, n).

    Returns:
        array: The array of rotated stress.
    """
    tensor_before = array2tensor(stress_before, 'stress', rank=3)
    tensor_after = np.einsum(
        'ijk,jlk,lmk->imk', r_matrix.transpose(1, 0, 2), tensor_before, r_matrix)
    stress_after = tensor2array(tensor_after, 'stress', rank=3)
    return stress_after


def tensor_cst(tensor, deg):
    """Coordinates System Transformer of tensor.

    Args:
        tensor (array): tensor of state_q
        deg (array): degree of rotation(counterclockwise is a plus number)

    Returns:
        array: tensor after transformed
    """
    R = R_matrix(deg, tensor.shape[0])
    tensor_cst = np.empty(tensor.shape)
    for i in range(0, tensor.shape[2]):
        if len(R.shape) == 3:
            tensor_cst[:, :, i] = R[:, :, i].T @ tensor[:, :, i] @ R[:, :, i]
        elif len(R.shape) == 2:
            tensor_cst[:, :, i] = R.T @ tensor[:, :, i] @ R

    return tensor_cst


def graph_params(
    title='',
    xlabel='',
    ylabel='',
    xticks='',
    yticks='',
    xlim='',
    ylim='',
    fontsize=18,
    labelsize=16,
    legend=False,
    anchor=None,
    legend_ncol=1,
    grid=False,
):
    """graph control

    Args:
        title (str, optional): graph title
        xlabel (str, optional): x axis label
        ylabel (str, optional): y axis label
        xticks (list[float], optional): ticks of x axis
        yticks (list[float], optional): ticks of y axis
        xlim (list[float], optional): range of x axis
        ylim (list[float], optional): range of y axis
        fontsize (int, optional): fontsize of title, xlabel and ylabel
        labelsize (int, optional): fontsize of ticks
        legend (int, optional): fontsize of legend
        anchor (tuple, optional): location of legend
        legend_ncol (int, optional): the number of columns of legend
        grid (bool, optional): to show grid or not
    """
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if len(xticks) > 0:
        plt.xticks(xticks)
    if len(yticks) > 0:
        plt.yticks(yticks)
    if len(xlim) == 2:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim) == 2:
        plt.ylim(ylim[0], ylim[1])
    plt.tick_params(labelsize=labelsize)
    if legend != False or anchor is not None:
        if legend in [True, False]:
            legend = 16
        if anchor is None:
            anchor = (1, 1)
        plt.legend(fontsize=legend, bbox_to_anchor=anchor, ncol=legend_ncol)
    if grid:
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999',
                 linestyle='-', alpha=0.2)


def calc_deform(node, disp, scale=1):
    """calculate deformed node coordinate.

    Args:
        node (array[float]): node coordinate
        disp (array[float]): node displacement
        scale (int or float, optional): scale coefficient, the default is 1

    Returns:    
        array: deformed node coordinate
    """
    assert len(node) == len(
        disp), 'the length of node and displacement are not agreed.'
    deform = node.copy()
    if node.shape[1] == 3:
        pass
    elif node.shape[1] == 4:
        node = node[:, 1:4]
    else:
        print_log(f'node data have {node.shape[1]} cols and not supported.')

    if disp.shape[1] == 3:
        pass
    elif disp.shape[1] == 4:
        disp = disp[:, 1:4]
    elif disp.shape[1] == 6:
        disp = disp[:, 0:3]
    elif disp.shape[1] == 7:
        disp = disp[:, 1:4]
    else:
        print_log(f'disp data have {disp.shape[1]} cols and not supported.')

    deform = node + disp * scale
    deform = np.hstack([np.arange(1, len(deform)+1).reshape((-1, 1)), deform])
    return deform


def contour_plot(points, values, scale=1, crange=(-100, 100)):
    """contour plot

    Args:
        points (array[float]): node or elem center coordinates
        values (array[float]): values
        scale (int or float, optional): scale coefficient of values, the default is 1
        crange (int, float, tuple, or list, optional): colorbar value range, the default, (-100, 100)
    """
    assert type(crange) in (int, float, tuple, list,
                            ), f'the type of crange {type(crange)} is not supported.'
    assert points.shape[1] in (
        3, 4), 'coordinates array should have 3 or 4 columns.'

    if type(crange) in (int, float):
        crange = (-crange, crange)
    else:
        if len(crange) == 0:
            raise Exception('crange is empty!')
        elif len(crange) == 1:
            crange = (-crange[0], crange[0])
        elif len(crange) >= 2:
            crange = (crange[0], crange[1])
    points = points if points.shape[1] == 3 else points[:, 1:4]

    plt.figure(figsize=(20, 6))
    plt.subplot(221)
    _ = np.argsort(-points[:, 1])
    plt.scatter(points[_, 0], points[_, 2], c=values[_]*scale,
                cmap=plt.cm.jet, norm=Normalize(crange[0], crange[1]))
    cbar = plt.colorbar()
    plt.axis('equal')
    plt.subplot(222)
    _ = np.argsort(points[:, 2])
    plt.scatter(points[_, 0], points[_, 1], c=values[_]*scale,
                cmap=plt.cm.jet, norm=Normalize(crange[0], crange[1]))
    cbar = plt.colorbar()
    plt.axis('equal')
    plt.subplot(223)
    _ = np.argsort(points[:, 1])
    plt.scatter(-points[_, 0], points[_, 2], c=values[_]*scale,
                cmap=plt.cm.jet, norm=Normalize(crange[0], crange[1]))
    cbar = plt.colorbar()
    plt.axis('equal')
    plt.subplot(224)
    _ = points[:, 2] == 0
    plt.scatter(points[_, 0], points[_, 1], c=values[_]*scale,
                cmap=plt.cm.jet, norm=Normalize(crange[0], crange[1]))
    cbar = plt.colorbar()
    plt.axis('equal')
    plt.show()


def tri_contour_plot(node, elem, value, crange=(-100, 100), colorbar=True):
    """contour plot of triangle mesh

    Args:
        node (np.array): node data
        elem (np.array): element data
        value (np.array): value which are defined in each element
        crange (tuple, optional): contour color range, the default, (-100, 100)
    """
    _elem_definition = []
    _values = []
    for _elem, _value in zip(elem[:, 1:5], value):
        _elem = list(_elem)
        if _elem[2] == _elem[3]:
            _elem_definition.append(_elem[:3])
            _values.append(_value)
        else:
            _elem_definition.append(_elem[:3])
            _elem_definition.append([_elem[0]]+_elem[2:])
            _values += [_value] * 2
    _elem_definition = np.array(_elem_definition) - 1

    triang = tri.Triangulation(node[:, 1], node[:, 2])
    triang.triangles = _elem_definition.astype('int32')
    plt.tripcolor(triang, _values, shading='flat', cmap=plt.cm.jet,
                  norm=Normalize(crange[0], crange[1]))
    if colorbar:
        cb = plt.colorbar()
        return cb


def create_plate(width, height, thickness, elem_type=4,
                 w_div='default', h_div='default',
                 grid_seed_x=[], grid_seed_y=[]):
    """create node and elem_data (iFEM input file) of plate

    Args:
        width (int): plate width
        height (int): plate height
        thickness (float): plate thickness
        elem_type (int): the number of nodes which are included in element
        w_div (int, optional): division number of width
        h_div (int, optional): division number of height
        grid_seed_x (list or array, optional): grid seed of x axis
        grid_seed_y (list or array, optional): grid seed of y axis

    Returns:
        array: node and elem data of plate

    Example:
        >>> width = 100
        >>> height = 50
        >>> thickness = 1
        >>> node_data, elem_data = create_plate(width, height, thickness)
    """
    assert elem_type in (3, 4), 'Specified element type is not defined.'
    w_div = int(width//10 + 1) if w_div == 'default' else w_div+1
    h_div = int(height//10 + 1) if h_div == 'default' else h_div+1

    x = np.linspace(0, width, w_div)
    y = np.linspace(0, height, h_div)
    if len(grid_seed_x) != 0:
        assert max(grid_seed_x) == width and min(
            grid_seed_x) == 0, f'the range of grid_seed_x should be (0, {width})'
        x = grid_seed_x
        w_div = len(grid_seed_x)
    if len(grid_seed_y) != 0:
        assert max(grid_seed_y) == height and min(
            grid_seed_y) == 0, f'the range of grid_seed_y should be (0, {height})'
        y = grid_seed_y
        h_div = len(grid_seed_y)

    x_coords, y_coords = np.meshgrid(x, y)
    node_data = np.vstack([np.arange(1, w_div*h_div+1),
                           x_coords.ravel(),
                           y_coords.ravel(),
                           np.zeros((w_div*h_div))])
    node_data = node_data.T

    elem_data = np.empty(((w_div-1)*(h_div-1), 7))
    for idx in range(1, len(elem_data)+1):
        row = idx - 1
        elem_data[row, 0] = idx
        elem_data[row, 1] = idx + (idx+(idx-1)//(w_div-1))//w_div
        elem_data[row, 2] = idx+1 + (idx+(idx-1)//(w_div-1))//w_div
        elem_data[row, 3] = idx+1+w_div + (idx+(idx-1)//(w_div-1))//w_div
        elem_data[row, 4] = idx+w_div + (idx+(idx-1)//(w_div-1))//w_div
    elem_data[:, 5] = thickness
    elem_data[:, 6] = 1  # mid surface

    if elem_type == 3:
        elem_ = np.sort(np.tile(elem_data, (2, 1)), axis=0)
        elem_[:, 0] = np.arange(1, len(elem_)+1)
        for i in range(len(elem_)):
            if i % 2 == 0:
                elem_[i, 4] = elem_data[i//2, 3]
            else:
                elem_[i, 1:3] = elem_data[i//2, 3:5]
                elem_[i, 3:5] = elem_data[i//2, 1]
        elem_data = elem_

    return node_data, elem_data


def create_square_pipe(length, width, height, thickness, elem_size=1):
    """create node and element data of square pipe

    Args:
        length (int): pipe length
        width (int): pipe width
        height (int): pipe height
        thickness (float): thickness of square pipe
        elem_size (int): length of one edge of elems

    Returns:
        array: node and elem data of square pipe
    """
    assert all([round(_ % elem_size, 10) == 0 for _ in (length, width, height)]), \
        'elem_size must be a common divisor of length, width, and height.'
    w_div = int(length / elem_size)
    h_div = int((2*(width+height)) / elem_size)
    node, elem = create_plate(length, 2*(width+height),
                              thickness, 4, w_div, h_div)
    _width = int(width / elem_size)
    _height = int(height / elem_size)
    new_node = np.vstack([
        np.vstack([np.arange(0, width, elem_size), np.zeros(_width)]).T,
        np.vstack([np.full(_height, width), np.arange(0, height, elem_size)]).T,
        np.vstack([np.arange(width, 0, -elem_size),
                  np.full(_width, height)]).T,
        np.vstack([np.zeros(_height), np.arange(height, 0, -elem_size)]).T,
        np.zeros(2).T
    ])
    node[:, 2:] = np.tile(new_node, (len(np.unique(node[:, 1])), 1, 1)).transpose(
        1, 0, 2).reshape((-1, 2))
    node_definition = elem[:, 1:5].copy()
    _remove = node_definition >= node[-(w_div+1), 0]
    node_definition[_remove] -= node[-(w_div+1), 0]-1
    elem[:, 1:5] = node_definition
    node = node[:-(w_div+1)]
    return node, elem


def create_pipe(length, diameter, thickness, r_div='default', l_div='default'):
    """create node and element data of square pipe

    Args:
        length (int): pipe length
        diameter ()
        thickness (float): thickness of square pipe
        elem_size (int): length of one edge of elems

    Returns:
        array: node and elem data of square pipe
    """
    node, elem = create_plate(length, round(
        diameter*np.pi, 3), thickness, 4, l_div, r_div)
    r_div = int(round(diameter*np.pi, 3)//10 +
                1) if r_div == 'default' else r_div+1
    l_div = int(length//10)+1 if l_div == 'default' else l_div+1
    theta = np.linspace(0, 2*np.pi, r_div)
    new_node = np.vstack([
        0.5 * diameter * np.cos(theta),
        0.5 * diameter * (np.sin(theta) + 1),
    ]).T
    node[:, 2:] = np.tile(new_node, (len(np.unique(node[:, 1])), 1, 1)).transpose(
        1, 0, 2).reshape((-1, 2))
    node_definition = elem[:, 1:5].copy()
    _remove = node_definition >= node[-l_div, 0]
    node_definition[_remove] -= node[-l_div, 0]-1
    elem[:, 1:5] = node_definition
    node = node[:-l_div]
    return node, elem


def elem_area(_, g1, g2, g3, node, elem):
    """Calculate each element area."""
    A = node[int(elem[_, g2])-1, 1:4] - node[int(elem[_, g1])-1, 1:4]
    B = node[int(elem[_, g3])-1, 1:4] - node[int(elem[_, g1])-1, 1:4]
    S = 0.5*np.sqrt((A[1]*B[2] - A[2]*B[1])**2 +
                    (A[2]*B[0] - A[0]*B[2])**2 + (A[0]*B[1] - A[1]*B[0])**2)
    return S


def calc_elem_S(node, elem):
    """Calculate element areas."""
    elem_S = np.empty(len(elem))
    for i, _ in enumerate(np.arange(0, len(elem))):
        if len(np.unique(elem[_, 1:5])) == 3:
            elem_S[i] = elem_area(_, 1, 2, 3, node, elem)
        else:
            elem_S[i] = elem_area(_, 1, 2, 3, node, elem) + \
                elem_area(_, 3, 1, 4, node, elem)
    return elem_S


def calc_elem_weight(elem, part_dict, mat_dict, elem_parts, elem_areas):
    """calculate element weights."""
    elem_weights = list()
    for i in range(len(elem)):
        _part = part_dict[elem_parts[i]]
        if _part['type'] == 'PSHELL':
            elem_weights.append(
                elem_areas[i]*_part['thickness']*mat_dict[_part['mat_id']]['density'])
        elif _part['type'] == 'PCOMP':
            _elem_weight = 0
            for v in _part['ply'].values():
                _elem_weight += elem_areas[i]*v['thickness'] * \
                    mat_dict[v['mat_id']]['density']
            elem_weights.append(_elem_weight)
        else:
            raise NotImplementedError(_part['type'] == 'PCOMP')
    return np.array(elem_weights)


def elem_strain_pcomp(output_file, case_start, case_num, elem_dict, verbal=True):
    with open(output_file, 'r') as f:
        output = f.readlines()

    for case in range(case_start, case_start+case_num):
        print_log('[strain] Case '+str(case)+' is starting...', verbal)
        strain = list()
        trigger_1 = False
        trigger_2 = False
        trigger_3 = False
        for i, row in enumerate(output):
            # Enter strain output part.
            if row.endswith('M A X I M U M   D I S P L A C E M E N T S\n'):
                trigger_1 = True
            # Enter target case output part.
            if trigger_1:
                if row.endswith('SUBCASE '+str(case)+'\n'):
                    trigger_2 = True
            # Exit target case output part.
            if trigger_2:
                if row[69:76] == 'SUBCASE':
                    if int(row[77:84]) != case:
                        break
            # Enter each strain value part.
            if trigger_1 & trigger_2:
                if len(row.split()) == 11:
                    try:
                        if int(row.split()[0]) in elem_dict.keys():
                            trigger_3 = True
                    except:
                        pass
            # Pass blank row.
            if trigger_3:
                if row == ' \n':
                    trigger_3 = False
            # Append strain data.
            if trigger_3:
                strain.append(row)
            # After blank row and headers.
            if trigger_1 & trigger_2:
                if len(row.split()) in (12, 14):
                    if row.split()[0] == 'ID':
                        trigger_3 = True

        # Convert format.
        strains = list()
        for row in strain:
            items = [float(item) for item in row.split()]
            if len(items) == 11:
                _elem_id = elem_dict[int(items[0])]
                items[0] = _elem_id
            elif len(items) == 10:
                items = [_elem_id] + items
            else:
                raise ValueError(row)
            strains.append(items)
        strains = np.array(strains)
        df = pd.DataFrame(strains)

        def extract_strain(df):
            mid = 2 if len(df) == 3 else 3 if len(df) == 5 else 0  # ply_id
            if mid == 0:
                print(df)
                raise ValueError(f'Unknown plies. => {len(df)}')
            row = sum([
                [df[2].iloc[j], df[3].iloc[j], 0., df[4].iloc[j], 0., 0.]
                for j in (mid-1, 0, -1)
            ], [])
            return row

        strains = df.iloc[:, 0:5].groupby(0).apply(extract_strain)
        strains = np.hstack([np.array(strains.index).reshape(-1, 1),
                            np.vstack([row for row in strains])])  # (n*19)

        if case == case_start:
            strain_all = strains.reshape(-1, 19, 1)
        else:
            strain_all = np.dstack([strain_all, strains.reshape(-1, 19, 1)])
    print_log('[strain] Successfully finished.', verbal)
    return strain_all


def print_log(message, verbal=True):
    """Print log message."""
    if verbal:
        print(f'[Info] {message}')


def import_subcase(file_path, verbal=True):
    """Import connection data between DLSA subcase and loadcase.

    Args:
        file_path (str): The path of subcase file (e.g. RSLTDLSA-SUBCASE.BDF).
    """
    with open(file_path, 'r') as f:
        rows = f.readlines()

    subcase2loadcase = {
        int(row.split()[-1]): int(rows[i+2].split()[-1]) for i, row in enumerate(rows) if 'SUBCASE' in row
    }
    print_log('Connections between subcase and loadcase are imported.', verbal)
    return subcase2loadcase


def import_loadcase(file_path, verbal=True):
    """Import connection data between DLSA loadcase and load ids.

    Args:
        file_path (str): The path of loadcase file (e.g. RSLTDLSA-LOADCASE.BDF).
    """
    with open(file_path, 'r') as f:
        rows = f.readlines()

    length = len(rows)
    loadcase2loadids = dict()
    for i, row in enumerate(rows):
        if 'LOAD' in row:
            row_i = row.split()
            loadcase2loadids[int(row_i[1])] = {
                int(row_i[4][:-1]): float(row_i[3])}
            for j in range(i+1, length):
                if 'LOAD' in rows[j]:
                    break
                row_j = rows[j].split()
                if row_j[0] == '*':
                    for k in range((len(row_j)-2)//2+1):
                        load_id = row_j[2*(k+1)]
                        load_id = int(
                            load_id[:-1]) if load_id[-1] == '+' else int(load_id)
                        loadcase2loadids[int(row_i[1])][load_id] = float(
                            row_j[2*k+1])
    print_log('Connections between loadcase and load_id are imported.', verbal)
    return loadcase2loadids


def import_pload(file_path, pload_dict, load_scales, verbal=True):
    """Import pressure load magnitude for each element.

    Args:
        file_path (str): The path to pressure load file (e.g. RSLTDLSA-SWLOADOUT.BDF).
        pload_dict (dict): The dict whose the key is elem_id and value is magnitude for each direction.
        load_scales (dict): The dict whose the key is load_id and the value is scale factor of the load_id.

    Example:
        >>> subcase = ni.import_subcase(dlsa_dir / 'RSLTDLSA-SUBCASE.BDF')
        >>> loadcase = ni.import_loadcase(dlsa_dir / 'RSLTDLSA-LOADCASE.BDF')
        >>> pload_dict = {elem_id: np.zeros(3) for elem_id in e_dict.keys()}  # The key is original elem_id.
        >>> k = 1  # subcase id to import pload.
        >>> for load_type in ('SWLOADIN', 'SWLOADOUT', 'DYNLOADIN', 'DYNLOADOUT'):
        >>>     pload_dict = ni.import_pload(dlsa_dir / f'RSLTDLSA-{load_type}.BDF', pload_dict, loadcase[subcase[k]])
        >>> pload_dict = {e_dict[k]: v for k, v in pload_dict.items()}  # The key is serial elem_id.
        >>> pload = np.vstack([v for v in pload_dict.values()])
    """
    try:
        with open(file_path, 'r') as f:
            rows = f.readlines()
        # Import pressure load for each element.
        load_ids = load_scales.keys()
        length = len(rows)
        for i, row in enumerate(rows):
            if 'PLOAD4' in row and int(row.split()[1]) in load_ids:
                row_ = row.split()
                if i+2 < length and rows[i+2][0] == '*' and len(rows[i+2]) >= 5:
                    row2_ = rows[i+2].split()
                    direction = np.array(
                        [float(row2_[j]) if j != 4 else float(row2_[j][:-1]) for j in range(2, 5)])
                else:
                    direction = np.array([0., 0., 1.])
                pload_dict[int(row_[2])] += float(row_[3]) * \
                    direction * load_scales[int(row_[1])]
        print_log(f'Pressure loads are imported from {file_path.name}', verbal)
    except FileNotFoundError as e:
        print(e)
    return pload_dict


class Data:
    """Data class."""

    def __init__(self, **kwargs):
        self.set_data(**kwargs)

    def set_data(self, **kwargs):
        """Set attribute."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def keys(self):
        """Return attribute list."""
        return [attr for attr in dir(self) if (attr[0] != '_') and (not callable(getattr(self, attr)))]


def import_model(input_path, renumber=True):
    try:
        with open(input_path) as f:
            model_data = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, encoding='Shift-JIS') as f:
            model_data = f.readlines()

    model = Data(input_path=input_path, renumber=renumber)

    part_dict = dict()
    mat_dict = dict()
    for i, row in enumerate(model_data):
        # Import part and thickness.
        if row[:6] == 'PSHELL':
            row = row.split()
            part_dict[int(row[1])] = {
                'type': 'PSHELL',
                'mat_id': int(row[2]),
                'thickness': abs(float(row[3])),
            }
        elif row[:5] == 'PCOMP':
            row = row.split()
            part_dict[int(row[1])] = {
                'type': 'PCOMP',
                'thickness': abs(float(row[2]))*2,
            }
            _plies_part_dict = dict()
            j = 0
            while True:
                try:
                    row_ = model_data[i+j+1].split()
                except IndexError:
                    break
                if row_[0] != '+':
                    break
                _plies_part_dict[j*2+1] = {
                    'mat_id': int(row_[1]),
                    'thickness': float(row_[2])
                }
                if len(row_) > 5:
                    _plies_part_dict[j*2+2] = {
                        'mat_id': int(row_[5]),
                        'thickness': float(row_[6])
                    }
                j += 1
            part_dict[int(row[1])]['ply'] = _plies_part_dict
        # Import materials.
        if row[:5] == 'MAT1 ':  # Isotropic Material
            mat_dict[int(row[8:16])] = {
                'type': 'MAT1',
                'youngs_modulus': _float_conv(row[16:24]),
                'poissons_ratio': _float_conv(row[32:40]),
                'density': _float_conv(row[40:48]),
            }
        elif row[:5] == 'MAT1*':  # Isotropic Material / alternate form
            row_2 = model_data[i+1]
            mat_dict[int(row[8:16])] = {
                'type': 'MAT1',
                'youngs_modulus': _float_conv(row[24:40]),
                'poissons_ratio': _float_conv(row[56:64]),
                'density': _float_conv(row_2[8:16]),
            }
        elif row[:5] == 'MAT2 ':  # Anisotropic Material
            mat_dict[int(row[8:16])] = {
                'type': 'MAT2',
                'g11': _float_conv(row[16:24]),
                'g12': _float_conv(row[24:32]),
                'g13': _float_conv(row[32:40]),
                'g22': _float_conv(row[40:48]),
                'g23': _float_conv(row[48:56]),
                'g33': _float_conv(row[56:64]),
                'density': _float_conv(row[64:72]),
            }
        elif row[:5] == 'MAT2*':  # Anisotropic Material / alternate form
            row_2 = model_data[i+1]
            mat_dict[int(row[8:24])] = {
                'type': 'MAT2',
                'g11': _float_conv(row[24:40]),
                'g12': _float_conv(row[40:56]),
                'g13': _float_conv(row[56:72]),
                'g22': _float_conv(row_2[8:24]),
                'g23': _float_conv(row_2[24:40]),
                'g33': _float_conv(row_2[40:56]),
                'density': _float_conv(row_2[56:72]),
            }
        elif row[:5] == 'MAT8 ':  # Orthotropic Material
            mat_dict[int(row[8:16])] = {
                'type': 'MAT8',
                'youngs_modulus_1': _float_conv(row[16:24]),
                'youngs_modulus_2': _float_conv(row[24:32]),
                'poissons_ratio': _float_conv(row[32:40]),
                'density': _float_conv(row[64:72]),
            }

    node = list()
    elem = list()
    elem_parts = list()
    for i in range(0, len(model_data)):
        # Import node coordinates.
        if model_data[i][:5] == 'GRID ':
            row = model_data[i]
            node.append([int(row[8:16]), _float_conv(row[24:32]),
                        _float_conv(row[32:40]), _float_conv(row[40:48])])
        elif model_data[i][:5] == 'GRID*':
            row = model_data[i].split()
            row_2 = model_data[i+1].split()
            try:
                node.append([int(row[1]), _float_conv(row[2]),
                            _float_conv(row[3]), _float_conv(row_2[1])])
            except:
                if len(row[2]) == 32:
                    node.append([int(row[1]), _float_conv(row[2][:16]), _float_conv(
                        row[2][16:]), _float_conv(row_2[1])])
                else:
                    node.append([int(row[1]), _float_conv(row[2][:15]), _float_conv(
                        row[2][15:]), _float_conv(row_2[1])])
        # Import element definitions.
        if model_data[i][:7] == 'CQUAD4*':
            row = model_data[i].split()
            row_2 = model_data[i+1].split()
            elem.append([int(row[1]), int(row[3]), int(row[4]), int(row_2[1]), int(
                row_2[2]), float(part_dict[int(row[2])]['thickness']), int(1)])
            elem_parts.append(int(row[2]))
        elif model_data[i][:7] == 'CQUAD4 ':
            row = model_data[i].split()
            elem.append([int(row[1]), int(row[3]), int(row[4]), int(row[5]), int(
                row[6]), float(part_dict[int(row[2])]['thickness']), int(1)])
            elem_parts.append(int(row[2]))
        elif model_data[i][:7] == 'CTRIA3 ':
            row = model_data[i].split()
            elem.append([int(row[1]), int(row[3]), int(row[4]), int(row[5]), int(
                row[5]), float(part_dict[int(row[2])]['thickness']), int(1)])
            elem_parts.append(int(row[2]))

    node = np.array(node)
    node = node[np.argsort(node[:, 0])]
    elem = np.array(elem)
    elem_parts = np.array(elem_parts)[np.argsort(elem[:, 0])]
    elem = elem[np.argsort(elem[:, 0])]

    # Remove nodes
    elem_definitions = np.unique(elem[:, 1:5])
    remove = [nid in elem_definitions for nid in node[:, 0]]
    node = node[remove]

    # Renumber node and element id if necessary.
    node_id = node[:, 0].copy().astype('int')
    elem_id = elem[:, 0].copy().astype('int')
    if renumber:
        node, elem = id_changer(node, elem)
    node_dict = dict(zip(node_id, node[:, 0].astype('int')))
    node_idict = dict(zip(node[:, 0].astype('int'), node_id))
    elem_dict = dict(zip(elem_id, elem[:, 0].astype('int')))
    elem_idict = dict(zip(elem[:, 0].astype('int'), elem_id))

    elem_c = calc_elem_center(node, elem)
    elem_areas = calc_elem_S(node, elem)
    elem_weights = calc_elem_weight(
        elem, part_dict, mat_dict, elem_parts, elem_areas)

    model.set_data(
        node=node,
        elem=elem,
        elem_c=elem_c,
        part_dict=part_dict,
        mat_dict=mat_dict,
        elem_parts=elem_parts,
        node_dict=node_dict,
        node_idict=node_idict,
        elem_dict=elem_dict,
        elem_idict=elem_idict,
        elem_areas=elem_areas,
        elem_weights=elem_weights,
    )
    return model


def _divider(kind):
    br = '$' + '\n'
    row = '$' + '*'*72 + '\n'
    blank = (72 - len(kind)) // 2
    message = '$' + ' '*blank + f'{kind}' + '\n'
    return br + row + message + row + br


def format_part(model):
    """Format property data."""
    if not hasattr(model, 'part_dict'):
        raise ValueError('"part_dict" is not defined in input model.')
    parts = str()
    for pid, _part in model.part_dict.items():
        parts += f'$ Property : Property{pid}\n'
        if _part['type'] == 'PSHELL':
            parts += p('PSHELL') + p(pid) + p(_part['mat_id']) + p(
                _part['thickness']) + p(_part['mat_id']) + '\n'
        elif _part['type'] == 'PCOMP':
            parts += p('PCOMP') + p(pid) + \
                p(-_part['thickness']/2) + p()*6 + '+\n'
            assert np.allclose(np.unique(
                np.diff(list(_part['ply'].keys()))), np.array([1])), 'Property error.'
            for ply_id, ply in _part['ply'].items():
                _ply = p(ply['mat_id']) + \
                    p(ply['thickness']) + p(0.) + p('YES')
                parts += p('+') + _ply if ply_id % 2 == 1 else _ply + '+\n'
            if ply_id % 2 == 0:
                parts = parts[:-2] + '\n'  # Remove last '+'.
            parts += '\n'
    return parts


def format_material(model):
    """Format material data."""
    if not hasattr(model, 'mat_dict'):
        raise ValueError('"mat_dict" is not defined in input model.')
    mats = str()
    for mid, _mat in model.mat_dict.items():
        mats += f'$ Material : Material{mid}\n'
        if _mat['type'] == 'MAT1':
            mats += p('MAT1*') + p(mid, 16) + p(_mat['youngs_modulus'], 16) + p(
                limit=16) + p(_mat['poissons_ratio'], 16) + '\n'
            mats += p('*') + p(_mat['density'], 16) + '\n'
        elif _mat['type'] == 'MAT2':
            mats += p('MAT2*') + p(mid, 16) + \
                p(_mat['g11'], 16) + p(_mat['g12'], 16) + \
                p(_mat['g13'], 16) + '\n'
            mats += p('*') + p(_mat['g22'], 16) + p(_mat['g23'], 16) + \
                p(_mat['g33'], 16) + p(_mat['density'], 16) + '\n'
        elif _mat['type'] == 'MAT8':
            raise NotImplementedError('')  # ToDo
    return mats


def format_node(model):
    if not hasattr(model, 'node'):
        raise ValueError('"node" is not defined in input model.')
    nodes = str()
    for n in model.node:
        nodes += p('GRID*') + p(int(n[0]), 16) + \
            p('', 16) + p(n[1], 16) + p(n[2], 16) + '\n'
        nodes += p('*') + p(n[3], 16) + '\n'
    return nodes


def format_elem(model):
    if not hasattr(model, 'elem'):
        raise ValueError('"elem" is not defined in input model.')
    if not hasattr(model, 'elem_parts'):
        raise ValueError('"elem_parts" is not defined in input model.')
    if hasattr(model, 'part_dict'):
        defined_parts = set(model.part_dict.keys())
        used_parts = set(model.elem_parts)
        if defined_parts != used_parts:
            if len(defined_parts - used_parts) > 0:
                print(
                    f'Part id {defined_parts - used_parts} is defined but not used.')
            if len(used_parts - defined_parts) > 0:
                raise ValueError(
                    f'Part id {defined_parts - used_parts} is not defined.')
    elems = str()
    for i, e in enumerate(model.elem):
        if len(set(model.elem[i, 1:5])) == 4:
            elems += p('CQUAD4') + p(int(e[0])) + p(int(model.elem_parts[i])) + ''.join(
                [p(int(e[j])) for j in range(1, 5)]) + '\n'
        else:
            elems += p('CTRIA3') + p(int(e[0])) + p(int(model.elem_parts[i])) + ''.join(
                [p(int(e[j])) for j in range(1, 4)]) + '\n'
    return elems


def create_model(file_path, model, spc=[], load=[], memo='', unit={}, subcase=1, subtitle=''):
    """Create Nastran input file.

    Args:
        file_path (str): File path to export.
        model (class): Model class including node, elem, part, mat and so on.
        spc (list): Single Point Constraint data. See "create_spc" function.
        load (list): Load data.
        memo (str): Memo for analysis.
        unit (dict): Analysis unit (e.g. {'Length': 'm', 'Time': 'sec'}).
        subcase (int): The number of subcase.
        subtitle (str): Subtitle of analysis.
    """
    discription = '$ ' + datetime.now().strftime('%Y/%m/%d %H:%M:%S') + \
        ' | ' + memo + '\n'
    unit_system = ''.join([f'$ {k}: {v}\n'for k, v in unit.items()])
    executive_control = '\n'.join([
        'SOL 101', 'CEND', 'ECHO = NONE', 'RESVEC = YES\n',
    ])
    if isinstance(subcase, int):
        subcase = subcase
    else:
        subcase = 1
        print(f'"{subcase}" is invalid as a subcase number. Set 1 instead.')
    subtitle = subtitle if subtitle != '' else 'static'
    case_control = '\n'.join([
        f'SUBCASE {subcase}', f'  SUBTITLE={subtitle}', '  SPC = 1000', '  LOAD = 2000',
        '  DISPLACEMENT(PRINT,REAL)=ALL', '  STRAIN(PRINT,REAL)=ALL', '  $STRESS(PRINT,REAL)=ALL\n',
    ])
    bulk_params = '\n'.join([
        'BEGIN BULK', 'PARAM    POST    -1',
        '$SUPORT,  1, 123456', '$PARAM, GRDPNT, 1', '$PARAM   INREL   -1\n',
    ])
    spc = ''.join(spc)
    load = ''.join(load)
    parts = format_part(model)
    mats = format_material(model)
    nodes = format_node(model)
    elems = format_elem(model)

    contents = [
        _divider('Discription'),
        discription,
        _divider('Unit System'),
        unit_system,
        _divider('Executive Control Section'),
        executive_control,
        _divider('Case Control Section'),
        case_control,
        _divider('Bulk Data Section'),
        bulk_params,
        _divider('Constraints of the Model'),
        spc,
        _divider('Loads of the Model'),
        load,
        _divider('Parts of the Model'),
        parts,
        _divider('Materials of the Model'),
        mats,
        _divider('Nones of the Model'),
        nodes,
        _divider('Elements of the Model'),
        elems,
        '$\nENDDATA\n$',
    ]
    with open(file_path, 'w') as f:
        f.write(''.join(contents)[1:])
