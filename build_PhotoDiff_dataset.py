from rdkit import Chem
from rdkit.Chem import Draw
import msgpack
import numpy as np
import json
import time
import os
from qm9.bond_analyze import get_bond_order

dataset_info = {'name': 'PhotoDiff',
                'atom_encoder': {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17},
                'atom_decoder': {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'},
                'colors_dic': {'H': (1,1,1), 'C': (0.7,0.7,0.7), 'N': (0,0,1), 'O': (1,0,0), 
                                'F': (0,1,0), 'S': (1,1,0), 'Cl': (0,1,1)},
                'radius_dic': {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, 
                                'S': 1.02, 'Cl': 0.99}
                }

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

def mol2smiles(mol):
    """
    Convert a molecule object to its corresponding SMILES representation.
    
    Args:
        mol: A molecule object.
        
    Returns:
        The SMILES representation of the molecule, or None if the molecule cannot be sanitized.
    """
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def build_molecule(sxyz, dataset_info):
    """
    Build a molecule object based on the given atomic coordinates and dataset information.

    Args:
        sxyz (numpy.ndarray): Atomic numbers and coordinates of the molecule.
        dataset_info (dict): Information about the dataset.

    Returns:
        mol (Chem.RWMol): The constructed molecule object.
    """
    atom_decoder = dataset_info['atom_decoder']
    X, A, E = build_xae_molecule(sxyz, dataset_info)
    print(X)
    print(E)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom])
        mol.AddAtom(a)

    all_bonds = np.argwhere(A)
    for bond in all_bonds:
        mol.AddBond(int(bond[0]), int(bond[1]), bond_dict[E[bond[0], bond[1]]])
    return mol

def build_xae_molecule(sxyz, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3
        atom_types: N
        returns:
        X: N         (int)  (atomic numbers)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond)
    """
    atom_decoder = dataset_info['atom_decoder']
    sxyz = np.array(sxyz)
    atom_types = sxyz[:, 0]
    positions = sxyz[:, 1:]
    num_atoms = len(atom_types)

    X = atom_types
    A = np.zeros((num_atoms, num_atoms), dtype=bool)
    E = np.zeros((num_atoms, num_atoms), dtype=int)

    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    for i in range(num_atoms):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], distances[i, j])
            if order > 0:
                A[i, j] = 1
                A[j, i] = 1
                E[i, j] = order
    return X, A, E

def read_sxyz_from_xyz_file(path, dataset_info):
    """
    Read atomic coordinates and symbols from an XYZ file and convert them to a nested list of sxyz format.

    Args:
        path (str): The path to the XYZ file.
        dataset_info (dict): A dictionary containing dataset information, including the atom encoder.

    Returns:
        list: A nested list of sxyz format, where each element includes atomic numbers and coordinates.

    """
    atom_encoder = dataset_info['atom_encoder']
    coords = np.loadtxt(path, skiprows=2, usecols=(1, 2, 3))
    atom_symbols = np.loadtxt(path, skiprows=2, usecols=0, dtype=str)
    atomic_numbers = np.array([atom_encoder[atom] for atom in atom_symbols])
    sxyz = np.hstack((atomic_numbers[:, None], coords))

    return sxyz.tolist()

def write_sxyz_to_xyz_file(sxyz, path, dataset_info):
    """
    Write the atomic coordinates in sxyz format to an XYZ file.

    Args:
        sxyz (list): List of atomic numers and coordinates.
        path (str): Path to the output XYZ file.
        dataset_info (dict): Dictionary containing dataset information.

    Returns:
        None
    """
    atom_decoder = dataset_info['atom_decoder']
    content = []
    for atom in sxyz:
        coords = [str(coord) for coord in atom[1:]]
        content.append([atom_decoder[atom[0]], *coords])

    np.savetxt(path, np.array(content), fmt='%s', delimiter=' ', header=f"{len(content)}\n", comments='')

def read_smiles_from_xyz_file(path):
    """
    Reads a XYZ file and returns the SMILES representation of the first molecule.

    Args:
        path (str): The path to the XYZ file.

    Returns:
        str: The SMILES representation of the first molecule in the XYZ file.
    """
    from openbabel import pybel

    mol = next(pybel.readfile("xyz", path))
    smi = mol.write(format="smi")

    return smi.split()[0].strip()

def find_sequence(sxyz, sequence, dataset_info):
    """
    Finds a connected sequence of atoms in a molecule based on their atomic symbols.

    Args:
        sxyz (str): The molecular geometry.
        sequence (list): The sequence of atomic numbers to search for.
        dataset_info (dict): Additional information about the dataset.

    Returns:
        tuple: A tuple containing a boolean indicating whether the sequence was found and 
        a list of atom indices representing the found sequence.
    """

    atom_encoder = dataset_info['atom_encoder']
    atomic_numbers, connection_matrix, _ = build_xae_molecule(sxyz, dataset_info)
    sequence = [atom_encoder[atom] for atom in sequence]

    def search(index, sequence, found_atom_indices):

        if index >= len(sequence):
            return True
        
        current_atom = sequence[index]
        
        for atom_index, atomic_number in enumerate(atomic_numbers):
            if atomic_number == current_atom and atom_index not in found_atom_indices:
                if index == 0 or connection_matrix[found_atom_indices[-1]][atom_index]:

                    found_atom_indices.append(atom_index)
                    found = search(index+1, sequence, found_atom_indices)

                    if found: 
                        return True
                    else:
                        found_atom_indices.pop()

        return False

    result =  []
    success = search(0, sequence, result)

    return success, result

def get_dihedral(u1, u2, u3, u4):
    """
    Calculate the dihedral angle between four points.

    Args:
        u1, u2, u3, u4: numpy arrays representing the four points.

    Returns:
        The dihedral angle in degrees, in the range [0, 180].
    """
    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1)**0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1)**0.5
    porm = np.sign((v1 * a3).sum(-1))
    rad = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)

    if not porm == 0:
        rad = rad * porm

    abs_deg = np.abs(rad * 180 / np.pi)

    return abs_deg

def add_index_to_sxyz(sxyz, index):
    """
    Add molecule index as the first column to the sxyz array.

    Parameters:
    sxyz (list): List of atomic number and coordinates.
    index (int): Molecule index.

    Returns:
    list: List of coordinates with the molecule index added as the first column.
    """
    sxyz = np.array(sxyz)
    index = np.ones((len(sxyz), 1)) * index

    return np.hstack((index, sxyz)).tolist()

def get_cnnc_coords(sxyz, dataset_info):
    """
    Get the coordinates of the CNNC sequence from the given sxyz.

    Parameters:
    sxyz (list): The sxyz list containing the atomic numbers and coordinates.
    dataset_info (dict): Information about the dataset.

    Returns:
    numpy.ndarray or None: The coordinates of the CNNC sequence if found, None otherwise.
    """
    sequence = ['C', 'N', 'N', 'C']
    success, result = find_sequence(sxyz, sequence, dataset_info)
    xyz = np.array(sxyz)[:, 1:]

    if success:
        return xyz[result]
    else:
        return None

def get_stereo_from_sxyz(sxyz, dataset_info):
    """
    Get the stereochemistry (E or Z) from the given sxyz coordinates.

    Args:
        sxyz (list): The sxyz coordinates.
        dataset_info (dict): Information about the dataset.

    Returns:
        str: The stereochemistry, either 'E' or 'Z'. Returns None if the cnnc_coords 
        are not available.
    """

    cnnc_coords = get_cnnc_coords(sxyz, dataset_info)

    if cnnc_coords is not None:
        return 'E' if get_dihedral(*cnnc_coords) > 90 else 'Z'
    else:
        return None

def remove_stereo_from_smiles(smiles):
    """
    Removes stereo information from a SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The SMILES string with stereo information removed.
    """
    if '/' in smiles:
        smiles = smiles.replace('/', '')
    if '\\' in smiles:
        smiles = smiles.replace('\\', '')
    if '[N][N]' in smiles:
        smiles = smiles.replace('[N][N]', 'N=N')

    return smiles

class DatasetProcessor:

    def __init__(self, dataset_info, write_xyz=False):
        self.data_dict = {"geom": [], "smiles": [], "num_atoms": [], "stereo": [], "energy": []}
        self.dataset_info = dataset_info
        self.geom_idx = 0
        self.write_xyz = write_xyz
        self.allowed_stereo = ['E', 'Z']

    def update_from_xyz_file(self, path):
        """
        Update the dataset from an XYZ file.

        Args:
            path (str): The path to the XYZ file.

        Returns:
            None
        """
        sxyz = read_sxyz_from_xyz_file(path, self.dataset_info)
        smiles = read_smiles_from_xyz_file(path)
        #stereo = get_stereo_from_sxyz(sxyz, self.dataset_info)
        stereo = "E"

        if stereo in self.allowed_stereo:
            self.add_mol(sxyz, smiles, stereo, 0)


    def update_from_mol_dict(self, mol_dict):
        """
        Updates the dataset with a new molecule.

        Args:
            mol_dict (dict): A dictionary containing the molecule information.

        Returns:
            None
        """
        sxyz = mol_dict["xyz"]
        num_atoms = len(sxyz)
        smiles = remove_stereo_from_smiles(mol_dict["species"]["smiles"])
        energy = mol_dict["props"]["totalenergy"]
        stereo = get_stereo_from_sxyz(sxyz, self.dataset_info)
        #print(stereo)

        idx = np.argwhere(np.array(self.data_dict['smiles'], dtype=str) == smiles).flatten()

        # molecule is not in dataset and is added
        if len(idx) == 0:
            if stereo in self.allowed_stereo:
                self.add_mol(sxyz, smiles, stereo, energy)

        # molecule is added if stereo is different or is replaced if energy is lower
        elif len(idx) == 1:
            if self.data_dict['stereo'][idx[0]] != stereo and stereo in self.allowed_stereo:
                self.add_mol(sxyz, smiles, stereo, energy)

            elif self.data_dict['stereo'][idx[0]] == stereo and energy < self.data_dict['energy'][idx[0]]:
                self.replace_mol_at_index(sxyz, energy, idx[0])

        # molecule is replaced if energy is lower
        elif len(idx) == 2:
            for i in idx:
                if self.data_dict['stereo'][i] == stereo and energy < self.data_dict['energy'][i]:
                    self.replace_mol_at_index(sxyz, energy, i)

        else:
            raise ValueError("More than two identical smiles in dataset.")

    def add_mol(self, sxyz, smiles, stereo, energy):
        """
        Add a molecule to the dataset.

        Args:
            sxyz (list): The molecular geometry in XYZ format.
            smiles (str): The SMILES representation of the molecule.
            stereo (str): The stereochemistry information of the molecule.
            energy (float): The energy of the molecule.

        Returns:
            None
        """
        self.data_dict['smiles'].append(smiles)
        for atm in add_index_to_sxyz(sxyz, self.geom_idx):
            self.data_dict['geom'].append(atm)
        self.data_dict['stereo'].append(stereo)
        self.data_dict['energy'].append(energy)
        self.data_dict['num_atoms'].append(len(sxyz))
        self.geom_idx += 1

        if self.write_xyz:
            write_sxyz_to_xyz_file(sxyz, f"./geoms/{self.geom_idx:06}.xyz", self.dataset_info)

    def replace_mol_at_index(self, sxyz, energy, idx):
        """
        Replaces the molecular geometry and energy at the specified index.

        Args:
            sxyz (list): The new molecular geometry in XYZ format.
            energy (float): The new energy value.
            idx (int): The index of the molecule to replace.

        Returns:
            None
        """
        self.data_dict['energy'][idx] = energy
        new_atoms = iter(add_index_to_sxyz(sxyz, idx))
        self.data_dict['geom'] = [next(new_atoms) if atom[0] == idx else atom for atom in self.data_dict['geom']]

        if self.write_xyz:
            write_sxyz_to_xyz_file(sxyz, f"./geoms/{self.geom_idx:06}.xyz", self.dataset_info)

    def save_dataset(self, base_path):
        """
        Save the dataset to disk.

        This method saves the dataset dictionary, geometries, and smiles to separate files on disk.
        The dataset dictionary is saved as a JSON file, geometries are saved as a NumPy array,
        and smiles are saved as a text file.

        Args:
            None

        Returns:
            None
        """
        data_dict_path = os.path.join(base_path, f"./{self.dataset_info['name']}.json")
        with open(data_dict_path, "w") as f:
            json.dump(self.data_dict, f)
        print(f"Dataset saved to {data_dict_path}.")

        geoms = np.array(self.data_dict['geom'])
        geoms_path = os.path.join(base_path, f"./{self.dataset_info['name']}.npy")
        np.save(geoms_path, geoms)
        print(f"Geometries saved to {geoms_path}.")

        smiles = self.data_dict['smiles']
        smiles_path = os.path.join(base_path, f"./{self.dataset_info['name']}_smiles.txt")
        with open(smiles_path, "w") as f:
            for s in smiles:
                f.write(s + "\n")
        print(f"Smiles saved to {smiles_path}.")

        num_atoms = self.data_dict['num_atoms']
        num_atoms_path = os.path.join(base_path, f"./{self.dataset_info['name']}_n.npy")
        np.save(num_atoms_path, num_atoms)
        print(f"Number of atoms saved to {num_atoms_path}.")


def process_dataset(path1, path2, max_packages=-1, report=10):
    """
    Process the dataset by reading data from a msgpack file and a directory of XYZ files.

    Args:
        path1 (str): Path to the msgpack file. Default is "./switches.msgpack".
        path2 (str): Path to the directory of XYZ files. Default is "./GEOMETRIES/".
        max_packages (int): Maximum number of packages to process. Default is -1, which means all packages will be processed.
        report (int): Number of packages to report progress. Default is 10.

    Returns:
        data_dict (DatasetProcessor): The processed dataset as a DatasetProcessor object.
    """

    assert os.path.exists(path1), f"File {path1} does not exist."
    assert os.path.exists(path2), f"Directory {path2} does not exist."

    num_geoms = 0
    geom_idx = 0
    num_exceptions = 0

    ########### Part 1 ###########

    unpacker = msgpack.Unpacker(open(path1, "rb"), strict_map_key=False)
    data_dict = DatasetProcessor(dataset_info, write_xyz=True)

    start = time.time()

    for i, sub_dict in enumerate(unpacker):

        print(f"Processing {i+1}th package...")
        if i % report == 0:
            print(f"Processed {num_geoms} geometries so far...")

        for j, (id, mol_dict) in enumerate(sub_dict.items()):

            try:

                # count all geometries
                num_geoms += 1

                # update dataset
                data_dict.update_from_mol_dict(mol_dict)

            except:

                # count exceptions
                num_exceptions += 1
                continue

        if max_packages > 0 and i == max_packages:
            break

    print(f"Part 1 done!")
    #print(f"Processed {i+1} packages with {((time.time() - start)/i):.2f} seconds per package.")
    #print(f"Processed {num_geoms} molecules and encountered {num_exceptions} exceptions.")

    ########### Part 2 ###########

    file_list = os.listdir(path2)

    for i, file in enumerate(file_list):
        
        try: 

            # count all geometries
            num_geoms += 1

            # update dataset
            data_dict.update_from_xyz_file(os.path.join(path2, file))
        
        except:

            # count exceptions
            num_exceptions += 1
            continue

    print(f"Part 2 done!")

    ##############################

    return data_dict


def eval_dataset(dataset_path, dataset_info):
    """
    Evaluate the dataset and return a dictionary with evaluation metrics.

    Args:
        path (str): The path to the dataset (npy file of the atoms).
        dataset_info (dict): A dictionary containing information about the dataset.

    Returns:
        dict: A dictionary with evaluation metrics, including the name of the dataset,
              the atom encoder, the atomic numbers, the atom decoder, the maximum number
              of nodes, the number of nodes per molecule, the atom types, the colors dictionary,
              the radius dictionary, and whether hydrogen atoms are included.
    """
    eval_dict = {'name': 'PhotoDiff',
                'atom_encoder': dataset_info['atom_encoder'],
                'atomic_nb': list(dataset_info['atom_encoder'].values()),
                'atom_decoder': list(dataset_info['atom_encoder'].keys()),
                'max_n_nodes': 0,
                'n_nodes': {},
                'atom_types': {},
                'colors_dic': list(dataset_info['colors_dic'].values()),
                'radius_dic': list(dataset_info['radius_dic'].values()),
                'with_h': True
                }

    atoms = np.load(dataset_path, allow_pickle=True)
    num_molecules = len(np.unique(atoms[:, 0]))
    mol_idx = 0
    num_atoms = 0

    for i, atom in enumerate(atoms):

        # count number of different atom types
        atom_type = atom[1]
        if atom_type in eval_dict['atom_types']:
            eval_dict['atom_types'][atom_type] += 1
        else:
            eval_dict['atom_types'][atom_type] = 1

        # count number of nodes per molecule
        if atom[0] == mol_idx:
            num_atoms += 1
        else:
            if num_atoms in eval_dict['n_nodes']:
                eval_dict['n_nodes'][num_atoms] += 1
            else:
                eval_dict['n_nodes'][num_atoms] = 1
            num_atoms = 0
            mol_idx = atom[0]

    # sort both dictionaries
    eval_dict['n_nodes'] = dict(sorted(eval_dict['n_nodes'].items()))
    eval_dict['atom_types'] = dict(sorted(eval_dict['atom_types'].items()))

    # find maximum number of nodes
    eval_dict['max_n_nodes'] = max(eval_dict['n_nodes'].keys())

    return eval_dict

def main(process=True, evaluate=False, test=False):
    """
    Main function for processing, evaluating, or testing the dataset.

    Args:
        process (bool): If True, process the dataset.
        evaluate (bool): If True, evaluate the dataset and save the config.
        test (bool): If True, perform testing on the dataset.

    Returns:
        None
    """

    # base path
    base_path = "./data/PhotoDiff/"

    # path to dataset 
    path1 = os.path.join(base_path, "switches.msgpack")
    path2 = os.path.join(base_path, "GEOMETRIES/")

    if process:

        # process dataset
        data_dict = process_dataset(path1, path2, max_packages=-1)
        data_dict.save_dataset(base_path)

    elif evaluate:

        # evaluate dataset and save config
        dataset_path = os.path.join(base_path, "PhotoDiff.npy")
        eval_dict = eval_dataset(dataset_path, dataset_info)
        config_path = os.path.join("./configs/", f"{dataset_info['name']}_config.json")
        with open(config_path, "w") as f:
            json.dump(eval_dict, f, sort_keys=False, indent=4, separators=(',', ': '))
        #print(eval_dict)

    elif test:

        # CAREFUL! Only for first time run:
        data = np.load(os.path.join(base_path, 'PhotoDiff_n.npy'), allow_pickle=True)
        perm = np.random.permutation(data.shape[0]).astype('int32')
        # print('Warning, currently taking a random permutation for '
        #       'train/val/test partitions, this needs to be fixed for'
        #       'reproducibility.')
        # assert not os.path.exists(os.path.join(base_path, 'geom_permutation.npy'))
        np.save(os.path.join(base_path, 'PhotoDiff_permutation.npy'), perm)


if __name__ == "__main__":
    main(process=False, evaluate=False, test=False)