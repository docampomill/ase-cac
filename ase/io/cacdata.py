import re
import numpy as np

from ase.atoms import Atoms
from ase.parallel import paropen
from ase.utils import basestring
from ase.calculators.lammps import Prism, convert


def read_cac_data(fileobj, Z_of_type=None, style="atomic",
                     sort_by_id=False, units="metal"):
    """Method which reads a LAMMPS data file.

    sort_by_id: Order the particles according to their id. Might be faster to
    switch it off.
    Units are set by default to the style=metal setting in LAMMPS.
    """
    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    # load everything into memory
    lines = f.readlines()

    # begin read_lammps_data
    comment = None
    N = None
    # N_types = None

    #==added by arman=====
    N_ELM= None
    N_ELM_types= None
    #========================
    xlo = None
    xhi = None
    ylo = None
    yhi = None
    zlo = None
    zhi = None
    xy = None
    xz = None
    yz = None
    pos_in = {}
    Node_vel_in={}
    #============added by arman
    # elem_id_nodes_in = {}
    # nodes_elemID_in = {}
    pos_elm_in = {}
    elm_node_type_in = {}
    elem_pos_in = {}
    #==================================
    travel_in = {}
    mol_id_in = {}
    charge_in = {}
    mass_in = {}
    vel_in = {}
    bonds_in = []
    angles_in = []
    dihedrals_in = []
    integration_in=[]
    interpolate_in=[]

    sections = [
        #=====added by Arman
        "Integration",
        "Interpolate",
        "Nodes",
        "Elements",
        "Node Velocities",
        #====================
        "Atoms",
        "Velocities",
        "Masses",
        "Charges",
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Impropers Pair Coeffs",
        "PairIJ Coeffs",
        "Pair Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ]
    header_fields = [
        "atoms",
        #====added by arman=====
        "elements",
        #==================
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "atom types",
        #==added by arman=======
        "element types",
        #==================
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
        "extra bond per atom",
        "extra angle per atom",
        "extra dihedral per atom",
        "extra improper per atom",
        "extra special per atom",
        "ellipsoids",
        "lines",
        "triangles",
        "bodies",
        "xlo xhi",
        "ylo yhi",
        "zlo zhi",
        "xy xz yz",
    ]
    sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
    header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"

    section = None
    header = True
    Node_VelTag=True
    # for number,line in enumerate(lines):
    for line in lines:

        if comment is None:
            comment = line.rstrip()
        else:
            line = re.sub("#.*", "", line).rstrip().lstrip()
            if re.match("^\\s*$", line):  # skip blank lines
                continue

        # check for known section names
        m = re.match(sections_re, line)
        if m is not None:
            section = m.group(0).rstrip().lstrip()
            header = False
            continue

        if header:
            field = None
            val = None
            # m = re.match(header_fields_re+"\s+=\s*(.*)", line)
            # if m is not None: # got a header line
            #   field=m.group(1).lstrip().rstrip()
            #   val=m.group(2).lstrip().rstrip()
            # else: # try other format
            #   m = re.match("(.*)\s+"+header_fields_re, line)
            #   if m is not None:
            #       field = m.group(2).lstrip().rstrip()
            #       val = m.group(1).lstrip().rstrip()
            m = re.match("(.*)\\s+" + header_fields_re, line)
            if m is not None:
                field = m.group(2).lstrip().rstrip()
                val = m.group(1).lstrip().rstrip()
            if field is not None and val is not None:

                # The 2 follwing lies have been commented off by Arman
                # if field == "atoms":
                #     N = int(val)

                # elif field == "atom types":
                #     N_types = int(val)

                if field == "elements":
                # elif field == "elements":
                    N_ELM = int(val)

                elif field == "element types":
                    N_ELM_type = int(val)

                elif field == "xlo xhi":
                    (xlo, xhi) = [float(x) for x in val.split()]
                elif field == "ylo yhi":
                    (ylo, yhi) = [float(x) for x in val.split()]
                elif field == "zlo zhi":
                    (zlo, zhi) = [float(x) for x in val.split()]
                elif field == "xy xz yz":
                    (xy, xz, yz) = [float(x) for x in val.split()]

        if section is not None:
            fields = line.split()


##==============added by arman==============================================
            if section == "Elements":  # id *
                id_elm = int(fields[0])
                if style == "cac" :
                    # element_type  x y z 
                    pos_elm_in[id_elm] = (
                        int(fields[1]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )      

                    elm_node_type_in[id_elm] = int(fields[2])
#======================================================================   
            elif section == "Nodes": # id *
                id = int(fields[5])    
                if style == "cac":
                    # id type nodes_elemID X Y Z 
                    pos_in[id] = (
                        int(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                        int(fields[0]),
                    )
                    #element_id
                    # nodes_elemID_in[id] = int(fields[0])    
#================================================================
                                        
            elif section == "Atoms":  # id *
                id = int(fields[0])
                if style == "full" and (len(fields) == 7 or len(fields) == 10):
                    # id mol-id type q x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[2]),
                        float(fields[4]),
                        float(fields[5]),
                        float(fields[6]),
                    )
                    mol_id_in[id] = int(fields[1])
                    charge_in[id] = float(fields[3])
                    if len(fields) == 10:
                        travel_in[id] = (
                            int(fields[7]),
                            int(fields[8]),
                            int(fields[9]),
                        )
                elif style == "atomic" and (
                        len(fields) == 5 or len(fields) == 8
                ):
                    # id type x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                    )
                    if len(fields) == 8:
                        travel_in[id] = (
                            int(fields[5]),
                            int(fields[6]),
                            int(fields[7]),
                        )
                elif (style in ("angle", "bond", "molecular")
                      ) and (len(fields) == 6 or len(fields) == 9):
                    # id mol-id type x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )
                    mol_id_in[id] = int(fields[1])
                    if len(fields) == 9:
                        travel_in[id] = (
                            int(fields[6]),
                            int(fields[7]),
                            int(fields[8]),
                        )
                elif style == "charge" and (len(fields) == 6 or len(fields) == 9):
                    # id type q x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[1]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )
                    charge_in[id] = float(fields[2])
                    if len(fields) == 9:
                        travel_in[id] = (
                            int(fields[6]),
                            int(fields[7]),
                            int(fields[8]),
                        )
                else:
                    raise RuntimeError(
                        "Style '{}' not supported or invalid "
                        "number of fields {}"
                        "".format(style, len(fields))
                    )
            elif section == "Velocities":  # id vx vy vz
                vel_in[int(fields[0])] = (
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                )
            elif section == "Masses":
                mass_in[int(fields[0])] = float(fields[1])

            #====added by arman ===================================
            
            elif section == "Node Velocities": # id *
                if Node_VelTag:
                    id_nd=0
                    # Num0=number
                    Node_VelTag = False

                id_nd+=1
                # element_id type X Y Z id

                # id_nd=number-Num0
                # print(id)

                Node_vel_in[id_nd] = (
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
            )


            elif section == "Interpolate":
                interpolate_in.append(
                    (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))
                )

            elif section == "Integration":
                integration_in.append(
                    (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))
                )
            #==================================================================================
          
            elif section == "Bonds":  # id type atom1 atom2
                bonds_in.append(
                    (int(fields[1]), int(fields[2]), int(fields[3]))
                )


            elif section == "Angles":  # id type atom1 atom2 atom3
                angles_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                    )
                )
            elif section == "Dihedrals":  # id type atom1 atom2 atom3 atom4
                dihedrals_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                        int(fields[5]),
                    )
                )

    # set cell
    cell = np.zeros((3, 3))
    cell[0, 0] = xhi - xlo
    cell[1, 1] = yhi - ylo
    cell[2, 2] = zhi - zlo
    if xy is not None:
        cell[1, 0] = xy
    if xz is not None:
        cell[2, 0] = xz
    if yz is not None:
        cell[2, 1] = yz

    box = np.zeros((9,))
    box[0] = xlo  
    box[1] = xhi
    box[2] = ylo
    box[3] = yhi
    box[4] = zlo
    box[5] = zhi
    if xy is not None:
        box[6] = xy
    if xz is not None:
        box[7] = xz
    if yz is not None:
        box[8] = yz
    #== added by Arman =========    
    N= len(pos_in)

    if len(pos_elm_in) > 0:
        N_elm= len(pos_elm_in)
    #===========================

    # initialize arrays for per-atom quantities
    positions = np.zeros((N, 3))

    if len(pos_elm_in) > 0:
        positions_elm = np.zeros((N_elm, 3))

    numbers = np.zeros((N), int)
    ids = np.zeros((N), int)
    types = np.zeros((N), int)
    if len(vel_in) > 0:
        velocities = np.zeros((N, 3))
    else:
        velocities = None
    if len(mass_in) > 0:
        masses = np.zeros((N))
    else:
        masses = None

    #======= added by arman =========
    if len(Node_vel_in) > 0:
        Node_velocities = np.zeros((N, 3))
    else:
        Node_velocities = None

    if len(interpolate_in) > 0:
        interpolates = np.zeros((4))
    else:
        interpolates = None

    if len(integration_in) > 0:
        integrations = np.zeros((4))
    else:
        integrations = None
  
    if style == "cac" :
        nodes_elemID = np.zeros((N), int)
    else:
        nodes_elemID = None

    if N_ELM is not None:
        numbers_elm = np.zeros((N_ELM), int)
    else:
        numbers_elm=None
    
    if N_ELM is not None:
        ids_elm = np.zeros((N_ELM), int)
    else:
        ids_elm=None

    if N_ELM is not None:
        types_elm = np.zeros((N_ELM), int)
    else:
        types_elm=None

    if len(elm_node_type_in) > 0:
        elm_node_type = np.zeros((N_ELM), int)
    else:
        elm_node_type = None
    
    #==========================================

    if len(mol_id_in) > 0:
        mol_id = np.zeros((N), int)
    else:
        mol_id = None
    if len(charge_in) > 0:
        charge = np.zeros((N), float)
    else:
        charge = None
    if len(travel_in) > 0:
        travel = np.zeros((N, 3), int)
    else:
        travel = None
    if len(bonds_in) > 0:
        bonds = [""] * N
    else:
        bonds = None
    if len(angles_in) > 0:
        angles = [""] * N
    else:
        angles = None
    if len(dihedrals_in) > 0:
        dihedrals = [""] * N
    else:
        dihedrals = None

    ind_of_id = {}
    # copy per-atom quantities from read-in values

    for (i, id) in enumerate(pos_in.keys()):
        # by id
        ind_of_id[id] = i
        if sort_by_id:
            ind = id - 1
        else:
            ind = i    
        type = pos_in[id][0]
        positions[ind, :] = [pos_in[id][1], pos_in[id][2], pos_in[id][3]]
        if velocities is not None:
            velocities[ind, :] = [vel_in[id][0], vel_in[id][1], vel_in[id][2]]

        #======= aqdded by arman ========== 
        if Node_velocities is not None:
            Node_velocities[ind, :] = [Node_vel_in[id][0], Node_vel_in[id][1], Node_vel_in[id][2]]

        #==================================

        if travel is not None:
            travel[ind] = travel_in[id]
        if mol_id is not None:
            mol_id[ind] = mol_id_in[id]
        if charge is not None:
            charge[ind] = charge_in[id]

        if nodes_elemID is not None:
            nodes_elemID[ind] = pos_in[id][4]

        ids[ind] = id
        # by type
        types[ind] = type
        if Z_of_type is None:
            numbers[ind] = type
        else:
            numbers[ind] = Z_of_type[type]
        if masses is not None:
            masses[ind] = mass_in[type]

    if len(pos_elm_in) > 0:
        for kk, r in enumerate(positions):
            elm_id_of_node=pos_in[kk+1][4]
            positions_elm[elm_id_of_node-1,:]+=r

        positions_elm=positions_elm/4

    ind_elm_of_id = {}
    # copy per-atom quantities from read-in values

    for (i_elm, id_elm) in enumerate(pos_elm_in.keys()):
        # by id
        ind_elm_of_id[id_elm] = i_elm
        if sort_by_id:
            ind_elm = id_elm - 1
        else:
            ind_elm = i_elm    
        type_elm  = pos_elm_in[id_elm][0]
        
        if elm_node_type is not None:
            elm_node_type[ind_elm] = elm_node_type_in[id_elm]

        ids_elm[ind_elm] = id_elm
        # by type
        types_elm[ind_elm] = type_elm

    if interpolates is not None:
        interpolates=[interpolate_in[0][0], interpolate_in[0][1], interpolate_in[0][2], interpolate_in[0][3]]

    if integrations is not None:
        integrations=[integration_in[0][0], integration_in[0][1], integration_in[0][2], integration_in[0][3]]
    
    # convert units
    positions = convert(positions, "distance", units, "ASE")
    cell = convert(cell, "distance", units, "ASE")
    if masses is not None:
        masses = convert(masses, "mass", units, "ASE")
    if velocities is not None:
        velocities = convert(velocities, "velocity", units, "ASE")

    if Node_velocities is not None:
        Node_velocities = convert(Node_velocities, "velocity", units, "ASE")
    # create ase.Atoms

    if len(pos_elm_in) > 0: 
        at = Atoms(
            positions=positions,
            positions_elm=positions_elm,
            numbers=numbers,
            masses=masses,
            interpolates=interpolates,
            integrations=integrations,
            nodes_elemID=nodes_elemID,
            elm_node_type=elm_node_type,
            types_elm=types_elm,
            cell=cell, box=box,
            pbc=[True, True, True],
        )
    else:
        at = Atoms(
            positions=positions,
            numbers=numbers,
            masses=masses,
            interpolates=interpolates,
            integrations=integrations,
            nodes_elemID=nodes_elemID,
            elm_node_type=elm_node_type,
            types_elm=types_elm,
            cell=cell, box=box,
            pbc=[True, True, True],
        )

    # set velocities (can't do it via constructor)
    if velocities is not None:
        at.set_velocities(velocities)
    at.arrays["id"] = ids
    at.arrays["type"] = types
    if travel is not None:
        at.arrays["travel"] = travel
    if mol_id is not None:
        at.arrays["mol-id"] = mol_id
    if charge is not None:
        at.arrays["initial_charges"] = charge
        at.arrays["mmcharges"] = charge.copy()

    if bonds is not None:
        for (type, a1, a2) in bonds_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            if len(bonds[i_a1]) > 0:
                bonds[i_a1] += ","
            bonds[i_a1] += "%d(%d)" % (i_a2, type)
        for i in range(len(bonds)):
            if len(bonds[i]) == 0:
                bonds[i] = "_"
        at.arrays["bonds"] = np.array(bonds)

    if angles is not None:
        for (type, a1, a2, a3) in angles_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            if len(angles[i_a2]) > 0:
                angles[i_a2] += ","
            angles[i_a2] += "%d-%d(%d)" % (i_a1, i_a3, type)
        for i in range(len(angles)):
            if len(angles[i]) == 0:
                angles[i] = "_"
        at.arrays["angles"] = np.array(angles)

    if dihedrals is not None:
        for (type, a1, a2, a3, a4) in dihedrals_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            i_a4 = ind_of_id[a4]
            if len(dihedrals[i_a1]) > 0:
                dihedrals[i_a1] += ","
            dihedrals[i_a1] += "%d-%d-%d(%d)" % (i_a2, i_a3, i_a4, type)
        for i in range(len(dihedrals)):
            if len(dihedrals[i]) == 0:
                dihedrals[i] = "_"
        at.arrays["dihedrals"] = np.array(dihedrals)

    at.info["comment"] = comment

    return at


def write_cac_data(fileobj, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, units="metal",
                      atom_style='atomic'):
    """Write atomic structure data to a LAMMPS data file."""
    if isinstance(fileobj, basestring):
        f = paropen(fileobj, "w", encoding="ascii")
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    # FIXME: We should add a check here that the encoding of the file object
    #        is actually ascii once the 'encoding' attribute of IOFormat objects
    #        starts functioning in implementation (currently it doesn't do
    #         anything).

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    f.write("{0} (CAC data file written by ASE) \n\n".format(f.name))

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    n_atoms1='0'
    f.write("{0} \t atoms \n".format(n_atoms1))

    if atom_style == 'cac':
        n_elem=len(atoms.get_types_elm())
        f.write("{0} \t elements \n".format(n_elem))

    def set_seq(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]
        

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        
        ### originally the next line was not commented: Arman Commneted it, because we are not intrested in sorting type alphabetically
        # species = sorted(set(symbols))
        species=set_seq(symbols)
        #species=symbols
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    f.write("{0}  \t atom types\n".format(n_atom_types))

    if atom_style == 'cac':

        elm_species=set_seq(atoms.get_types_elm())
        n_elm_types = len(elm_species)
        f.write("{0} \t element types\n".format(n_elm_types))

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    f.write("\n")


    #shifted box

    # #Get cell parameters and convert from ASE units to LAMMPS units
    # xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
    #         "ASE", units)

    # f.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    # f.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    # f.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    ##### =====  added by Arman (fixed box )================
    BOX=atoms.get_box()

    xlo=BOX[0]   
    xhi=BOX[1] 
    ylo=BOX[2] 
    yhi=BOX[3] 
    zlo=BOX[4] 
    zhi=BOX[5] 
    xy=BOX[6] 
    xz=BOX[7] 
    yz=BOX[8] 

    f.write("{0:1.17g} {1:2.17g} xlo xhi\n".format(xlo, xhi))
    f.write("{0:1.17g} {1:2.17g} ylo yhi\n".format(ylo, yhi))
    f.write("{0:1.17g} {1:2.17g} zlo zhi\n".format(zlo, zhi))
    #==================================================

    if force_skew or p.is_skewed():
        f.write(
            "{0:1.17g} {1:2.17g} {2:3.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
#============================ Masses
    if atom_style == 'cac':
        Masses=[]
        for jj in range(len(species)):
            Masses.append(atoms.get_masses()[jj])

        f.write("\n")
        f.write("Masses \n\n")    

        for kk in range(len(Masses)):
            f.write("{0:1.5g} {1:2.5g} \n".format(kk+1, Masses[kk]))
#================================

    if atom_style == 'cac':
        Interpolate=[]
        for jj in range(len(atoms.get_interpolates())):
            Interpolate.append(atoms.get_interpolates()[jj])

        f.write("\n")
        f.write("Interpolate \n")    
        f.write("\n")
        for jj in range(len(Interpolate)):
            f.write("{0:1.5g} ".format(Interpolate[jj]))

#================================
    if atom_style == 'cac':
        Integration=[]
        for jj in range(len(atoms.get_integrations())):
            Integration.append(atoms.get_integrations()[jj])

        f.write("\n\n")
        f.write("Integration \n") 
        f.write("\n")   

        for jj in range(len(Integration)):
            f.write("{0:1.5g} ".format(Integration[jj]))
#================================



    #======= one source of issue#==== changing in the perodic boundary direction to keep all atoms inside the box ======
###======= fixed positions (not shifted)=========    
    # pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)

    # if atom_style == 'atomic':
    #     for i, r in enumerate(pos):
    #         # Convert position from ASE units to LAMMPS units
    #         r = convert(r, "distance", "ASE", units)
    #         s = species.index(symbols[i]) + 1
    #         f.write(
    #             "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
    #                 *(i + 1, s) + tuple(r)
    #             )
    #         )



###======= fixed positions (not shifted)=========
     # pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)
    pos = atoms.get_positions()

    if atom_style == 'atomic':
        f.write("\n\n")
        f.write("Atoms \n\n")
        for i, r in enumerate(pos):

            # # Convert position from ASE units to LAMMPS units
            # r = convert(r, "distance", "ASE", units)  
                      
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>1} {1:>1} {2:1.17g} {3:1.17g} {4:1.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    if atom_style == 'cac':
        f.write("\n\n")
        f.write("Elements \n\n")
        no_elm=max(atoms.get_nodes_elemID())
        elm_pos_sum= np.zeros((no_elm, 3))
        elm_id=[]
        for i, r in enumerate(pos):
            elm_id_of_node=atoms.get_nodes_elemID()[i]
            # # Convert position from ASE units to LAMMPS units
            # r = convert(r, "distance", "ASE", units)
            elm_pos_sum[elm_id_of_node-1,:]+=r


        elm_pos=elm_pos_sum/4
        for jj in range (no_elm):
            f.write(
                "{0:>1} {1:>1} {2:1.17g} {3:1.17g} {4:1.17g} {5:>1}\n".format(
                    *(int(jj+1), atoms.get_types_elm()[jj], atoms.get_elm_node_type()[jj]) + tuple(elm_pos[jj]) 
                )
            )

        f.write("\n\n")
        f.write("Nodes \n\n")
        for i, r in enumerate(pos):

            # # Convert position from ASE units to LAMMPS units
            # r = convert(r, "distance", "ASE", units)  

            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>1} {1:>1} {2:1.17g} {3:1.17g} {4:1.17g} {5:>1}\n".format(
                    *(atoms.get_nodes_elemID()[i], s) + tuple(r) , (i + 1)
                )
            )
    elif atom_style == 'charge':
        f.write("\n\n")
        f.write("Atoms \n\n")        
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n".format(
                    *(i + 1, s, q) + tuple(r)
                )
            )
    elif atom_style == 'full':
        f.write("\n\n")
        f.write("Atoms \n\n")        
        charges = atoms.get_initial_charges()
        molecule = 1 # Assign all atoms to a single molecule
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} {6:23.17g}\n".format(
                    *(i + 1, molecule, s, q) + tuple(r)
                )
            )
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        f.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            f.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    f.flush()
    if close_file:
        f.close()
