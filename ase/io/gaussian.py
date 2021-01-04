import re
import warnings
from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.gaussian import Gaussian

from ase.data import atomic_numbers
from ase.data.isotopes import download_isotope_data

_link0_keys = [
    'mem',
    'chk',
    'oldchk',
    'schk',
    'rwf',
    'oldmatrix',
    'oldrawmatrix',
    'int',
    'd2e',
    'save',
    'nosave',
    'errorsave',
    'cpu',
    'nprocshared',
    'gpucpu',
    'lindaworkers',
    'usessh',
    'ssh',
    'debuglinda',
]


_link0_special = [
    'kjob',
    'subst',
]


# Certain problematic methods do not provide well-defined potential energy
# surfaces, because these "composite" methods involve geometry optimization
# and/or vibrational frequency analysis. In addition, the "energy" calculated
# by these methods are typically ZPVE corrected and/or temperature dependent
# free energies.
_problem_methods = [
    'cbs-4m', 'cbs-qb3', 'cbs-apno',
    'g1', 'g2', 'g3', 'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp4',
    'w1', 'w1u', 'w1bd', 'w1ro',
]


_xc_to_method = dict(
    pbe='pbepbe',
    pbe0='pbe1pbe',
    hse06='hseh1pbe',
    hse03='ohse2pbe',
    lda='svwn',  # gaussian "knows about" LSDA, but maybe not LDA.
    tpss='tpsstpss',
    revtpss='revtpssrevtpss',
)


def write_gaussian_in(fd, atoms, properties=None, **params):
    params = deepcopy(params)

    if properties is None:
        properties = ['energy']

    # pop method and basis and output type
    method = params.pop('method', None)
    basis = params.pop('basis', None)
    fitting_basis = params.pop('fitting_basis', None)
    output_type = '#{}'.format(params.pop('output_type', 'P'))
    if output_type == '#':
        output_type = '#P'

    # basisfile, only used if basis=gen
    basisfile = params.pop('basisfile', None)

    # basis can be omitted if basisfile is provided
    if basisfile is not None and basis is None:
        basis = 'gen'

    # determine method from xc if it is provided
    if method is None:
        xc = params.pop('xc', None)
        if xc is not None:
            method = _xc_to_method.get(xc.lower(), xc)

    # If the user requests a problematic method, rather than raising an error
    # or proceeding blindly, give the user a warning that the results parsed
    # by ASE may not be meaningful.
    if method is not None:
        if method.lower() in _problem_methods:
            warnings.warn(
                'The requested method, {}, is a composite method. Composite '
                'methods do not have well-defined potential energy surfaces, '
                'so the energies, forces, and other properties returned by '
                'ASE may not be meaningful, or they may correspond to a '
                'different geometry than the one provided. '
                'Please use these methods with caution.'.format(method)
            )

    # determine charge from initial charges if not passed explicitly
    charge = params.pop('charge', None)
    if charge is None:
        charge = atoms.get_initial_charges().sum()

    # determine multiplicity from initial magnetic moments
    # if not passed explicitly
    mult = params.pop('mult', None)
    if mult is None:
        mult = atoms.get_initial_magnetic_moments().sum() + 1

    # pull out raw list of explicit keywords for backwards compatibility
    extra = params.pop('extra', None)

    # pull out any explicit IOPS
    ioplist = params.pop('ioplist', None)

    # also pull out 'addsec', which e.g. contains modredundant info
    addsec = params.pop('addsec', None)

    # set up link0 arguments
    out = []
    for key in _link0_keys:
        if key not in params:
            continue
        val = params.pop(key)
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append('%{}'.format(key))
        else:
            out.append('%{}={}'.format(key, val))

    # These link0 keywords have a slightly different syntax
    for key in _link0_special:
        if key not in params:
            continue
        val = params.pop(key)
        if not isinstance(val, str) and isinstance(val, Iterable):
            val = ' '.join(val)
        out.append('%{} L{}'.format(key, val))

    # begin route line
    # note: unlike in old calculator, each route keyword is put on its own
    # line.
    if basis and method and fitting_basis:
        out.append('{} {}/{}/{} ! ASE formatted method and basis'
                   .format(output_type, method, basis, fitting_basis))
    elif basis and method:
        out.append('{} {}/{} ! ASE formatted method and basis'
                   .format(output_type, method, basis))
    else:
        output_string = '{}'.format(output_type)
        for value in [method, basis]:
            if value is not None:
                output_string += ' {}'.format(value)
        out.append(output_string)

    for key, val in params.items():
        # assume bare keyword if val is falsey, i.e. '', None, False, etc.
        # also, for backwards compatibility: assume bare keyword if key and
        # val are the same
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append(key)
        elif not isinstance(val, str) and isinstance(val, Iterable):
            out.append('{}({})'.format(key, ','.join(val)))
        else:
            out.append('{}({})'.format(key, val))

    if ioplist is not None:
        out.append('IOP(' + ', '.join(ioplist) + ')')

    if extra is not None:
        out.append(extra)

    # Add 'force' iff the user requested forces, since Gaussian crashes when
    # 'force' is combined with certain other keywords such as opt and irc.
    if 'forces' in properties and 'force' not in params:
        out.append('force')

    # header, charge, and mult
    out += ['', 'Gaussian input prepared by ASE', '',
            '{:.0f} {:.0f}'.format(charge, mult)]

    # atomic positions
    for i, atom in enumerate(atoms):
        # this formatting was chosen for backwards compatibility reasons, but
        # it would probably be better to
        # 1) Ensure proper spacing between entries with explicit spaces
        # 2) Use fewer columns for the element
        # 3) Use 'e' (scientific notation) instead of 'f' for positions
        if atoms.arrays['gaussian_info'][i] is not None:
            symbol_section = atom.symbol + \
                '(' + atoms.arrays['gaussian_info'][i] + ')'
            out.append('{:<10s}{:20.10f}{:20.10f}'
                       '{:20.10f}'.format(symbol_section,
                                          *atom.position))
        else:
            out.append('{:<10s}{:20.10f}{:20.10f}{:20.10f}'
                       .format(atom.symbol, *atom.position))

    # unit cell vectors, in case of periodic boundary conditions
    for ipbc, tv in zip(atoms.pbc, atoms.cell):
        if ipbc:
            out.append('TV {:20.10f}{:20.10f}{:20.10f}'.format(*tv))

    out.append('')

    # if basis='gen', set basisfile. Either give a path to a basisfile, or
    # read in the provided file and paste it verbatim
    if basisfile is not None:
        if basisfile[0] == '@':
            out.append(basisfile)
        else:
            with open(basisfile, 'r') as f:
                out.append(f.read())
    else:
        if basis is not None and basis.lower() == 'gen':
            raise InputError('Please set basisfile')

    if addsec is not None:
        out.append('')
        if isinstance(addsec, str):
            out.append(addsec)
        elif isinstance(addsec, Iterable):
            out += list(addsec)

    out += ['', '']
    fd.write('\n'.join(out))


# Regexp for reading an input file:

_re_link0 = re.compile(r'^\s*%([^\=\)\(!]+)=?([^\=\)\(!]+)?(!.+)?')
# Link0 lines are in the format:
# '% keyword = value' or '% keyword'
# (with or without whitespaces)

_re_output_type = re.compile(r'^\s*#\s*([NPTnpt]?)\s*')
# The start of the route section begins with a '#', and then may
# be followed by the desired level of output in the output file: P, N or T.

_re_method_basis = re.compile(
    r"\s*([\w-]+)\s*\/([^/=!]+)([\/]([^!]+))?\s*(!.+)?")
# Matches method, basis and optional fitting basis in the format:
# method/basis/fitting_basis ! comment
# They will appear in this format if the Gaussian file has been generated
# by ASE using a calculator with the basis and method keywords set.

_re_chgmult = re.compile(r'^\s*[+-]?\d+(?:,\s*|\s+)[+-]?\d+\s*$')
# This is a bit more complex of a regex than we typically want, but it
# can be difficult to determine whether a line contains the charge and
# multiplicity, rather than just another route keyword. By making sure
# that the line contains exactly two *integers*, separated by either
# a comma (and possibly whitespace) or some amount of whitespace, we
# can be more confident that we've actually found the charge and multiplicity.


class GaussianConfiguration:

    def __init__(self, atoms, parameters):
        self.atoms = atoms
        self.parameters = parameters

    def get_atoms(self):
        return self.atoms

    def get_parameters(self):
        return self.parameters

    def get_calculator(self):
        self.calc = Gaussian(atoms=self.atoms)
        self.calc.parameters = self.parameters
        return self.calc

    @staticmethod
    def parse_gaussian_input(gaussian_input):
        '''Reads a gaussian input file into an atoms object and
        parameters dictionary.

        Parameters
        ----------
        gaussian_input
            An open gaussian input file

        Returns
        ---------
        GaussianConfiguration
            Contains an atoms object created using the structural
            information from the input file.
            Contains a parameters dictionary, which stores any
            keywords and options found in the link-0 and route
            sections of the input file.
        '''
        parameters = {}
        route_section = False
        atoms_section = False
        atom_masses = []
        atoms_info = []
        symbols = []
        positions = []
        pbc = np.zeros(3, dtype=bool)
        cell = np.zeros((3, 3))
        npbc = 0
        count_iso = 0
        atoms_saved = False
        readiso = False

        for line in gaussian_input:
            link0_match = _re_link0.match(line)
            output_type_match = _re_output_type.match(line)
            chgmult_match = _re_chgmult.match(line)
            method_basis_match = _re_method_basis.match(line)
            # The first blank line appears at the end of the route section
            # and a blank line appears at the end of the atoms section
            if line == '\n':
                route_section = False
                atoms_section = False
            elif link0_match:
                value = link0_match.group(2)
                if value is not None:
                    value = value.strip()
                parameters.update({link0_match.group(1).lower().strip():
                                   value})
            elif output_type_match and not route_section:
                route_section = True
                # remove #_ ready for looking for method/basis/parameters:
                line = line.strip(output_type_match.group(0))
                method_basis_match = _re_method_basis.match(line)
                parameters.update({'output_type': output_type_match.group(1)})
            elif chgmult_match:
                chgmult = chgmult_match.group(0).split()
                parameters.update(
                    {'charge': int(chgmult[0]), 'mult': int(chgmult[1])})
                # After the charge and multiplicty have been set, the atoms
                # section of the input file begins:
                atoms_section = True
            elif atoms_section:
                if (line.split()):
                    # reads any info in parantheses after the atom symbol
                    # and stores it in atoms_info as a dict:
                    atom_info_match = re.search(r'\(([^\)]+)\)', line)
                    if atom_info_match:
                        line = line.replace(atom_info_match.group(0), '')
                        tokens = line.split()
                        symbol = tokens[0]
                        atom_info = atom_info_match.group(1)
                        atom_info_dict = GaussianConfiguration \
                            .get_route_params(atom_info_match.group(1))
                        atom_info_dict = {k.lower(): v for k, v
                                          in atom_info_dict.items()}
                        atom_mass = atom_info_dict.get('iso', None)
                        if atom_mass is not None:
                            if atom_mass.isnumeric():
                                # will be true if atom_mass is integer
                                try:
                                    atom_mass = download_isotope_data(
                                    )[atomic_numbers[symbol]][
                                        int(atom_mass)]['mass']
                                    atom_info_dict['iso'] = str(atom_mass)
                                except KeyError:
                                    pass
                        atom_info = ""
                        for key, value in atom_info_dict.items():
                            atom_info += key + '=' + value + ', '
                        atom_info = atom_info.strip(', ')
                    else:
                        tokens = line.split()
                        symbol = tokens[0]
                        atom_info = None
                        atom_mass = None
                    # try:
                    pos = list(map(float, tokens[1:4]))
                    # except ValueError:
                    #     print("Error in molecule specification in gaussian input file.")
                    if symbol.upper() == 'TV':
                        pbc[npbc] = True
                        cell[npbc] = pos
                        npbc += 1
                    else:
                        symbols.append(symbol)
                        positions.append(pos)
                        atoms_info.append(atom_info)
                        atom_masses.append(atom_mass)

                    atoms_saved = True
            elif atoms_saved:  # we must be after the atoms section
                if count_iso == 0:
                    freq_options = parameters.get('freq', None)
                    if freq_options:
                        freq_name = 'freq'
                    else:
                        freq_options = parameters.get('frequency', None)
                        freq_name = 'frequency'
                    if freq_options is not None:
                        freq_options = freq_options.lower()
                        if 'readiso' or 'readisotopes' in freq_options:
                            if 'readisotopes' in freq_options:
                                iso_name = 'readisotopes'
                            else:
                                iso_name = 'readiso'
                            # print(freq_options.split(','))
                            freq_options = [v.group() for v in re.finditer(
                                r'[^\,/\s]+', freq_options)]
                            freq_options.remove(iso_name)
                            new_freq_options = ''
                            for v in freq_options:
                                new_freq_options += v + ' '
                            parameters[freq_name] = new_freq_options
                            readiso = True
                            atom_masses = []
                            # when count_iso is 0 we are in the line where
                            # temperature, pressure, [scale] is saved
                            line = line.replace(
                                '[', '').replace(']', '').split('!')[0]
                            tokens = line.strip().split()
                            try:
                                parameters.update({'temperature': tokens[0]})
                                parameters.update({'pressure': tokens[1]})
                                parameters.update({'scale': tokens[2]})
                            except IndexError:
                                pass
                elif readiso:
                    try:
                        atom_masses.append(float(line.split('!')[0]))
                    except ValueError:
                        atom_masses.append(None)
                count_iso += 1

            if route_section:
                if method_basis_match:
                    ase_gen_comment = '! ASE formatted method and basis'
                    if method_basis_match.group(5) == ase_gen_comment:
                        parameters.update(
                            {'method': method_basis_match.group(1)})
                        parameters.update(
                            {'basis': method_basis_match.group(2)})
                        if method_basis_match.group(4):
                            parameters.update(
                                {'fitting_basis': method_basis_match.group(4)})
                        continue

                parameters.update(GaussianConfiguration.get_route_params(line))

        if readiso:
            if len(atom_masses) < len(symbols):
                for i in range(0, len(symbols) - len(atom_masses)):
                    atom_masses.append(None)
        atoms = Atoms(symbols, positions, pbc=pbc,
                      cell=cell, masses=atom_masses)
        atoms.new_array('gaussian_info', np.array(atoms_info))
        return GaussianConfiguration(atoms, parameters)

    @staticmethod
    def get_route_params(line):
        '''Reads a line of the route section of a gaussian input file.

        Parameters
        ----------
        line (string)
            A line of the route section of a gaussian input file.

        Returns
        ---------
        params (dict)
            Contains the keywords and options found in the line.
        '''
        params = {}
        line = line.strip(' #')
        line = line.split('!')[0]  # removes any comments
        # First, get the keywords and options sepatated with
        # parantheses:
        match_iterator = re.finditer(r'\(([^\)]+)\)', line)
        index_ranges = []
        for match in match_iterator:
            index_range = [match.start(0), match.end(0)]
            options = match.group(1)
            # keyword is last word in previous substring:
            keyword_string = line[:match.start(0)]
            keyword_match_iter = [k for k in re.finditer(
                r'[^\,/\s]+', keyword_string) if k.group() != '=']
            keyword = keyword_match_iter[-1].group().strip(' =')
            index_range[0] = keyword_match_iter[-1].start()
            params.update({keyword: options})
            index_ranges.append(index_range)

        # remove from the line the keywords and options that we have saved:
        index_ranges.reverse()
        for index_range in index_ranges:
            start = index_range[0]
            stop = index_range[1]
            line = line[0: start:] + line[stop + 1::]

        # Next, get the keywords and options separated with
        # an equals sign, and those without an equals sign
        # must be keywords without options:

        # remove any whitespaces around '=':
        line = re.sub(r'\s*=\s*', '=', line)
        line = [x for x in re.split(r'[\s,\/]', line) if x != '']

        for s in line:
            if '=' in s:
                s = s.split('=')
                keyword = s.pop(0)
                options = s.pop(0)
                for string in s:
                    options += '=' + string
                params.update({keyword: options})
            else:
                if len(s) > 0:
                    params.update({s: None})

        return params


def read_gaussian_in(fd, get_calculator=False):
    gaussian_input = GaussianConfiguration.parse_gaussian_input(fd)
    atoms = gaussian_input.get_atoms()

    if get_calculator:
        atoms.calc = gaussian_input.get_calculator()

    return atoms


# In the interest of using the same RE for both atomic positions and forces,
# we make one of the columns optional. That's because atomic positions have
# 6 columns, while forces only has 5 columns. Otherwise they are very similar.
_re_atom = re.compile(
    r'^\s*\S+\s+(\S+)\s+(?:\S+\s+)?(\S+)\s+(\S+)\s+(\S+)\s*$'
)
_re_forceblock = re.compile(r'^\s*Center\s+Atomic\s+Forces\s+\S+\s*$')
_re_l716 = re.compile(r'^\s*\(Enter .+l716.exe\)$')


def _compare_merge_configs(configs, new):
    """Append new to configs if it contains a new geometry or new data.

    Gaussian sometimes repeats a geometry, for example at the end of an
    optimization, or when a user requests vibrational frequency
    analysis in the same calculation as a geometry optimization.

    In those cases, rather than repeating the structure in the list of
    returned structures, try to merge results if doing so doesn't change
    any previously calculated values. If that's not possible, then create
    a new "image" with the new results.
    """
    if not configs:
        configs.append(new)
        return

    old = configs[-1]

    if old != new:
        configs.append(new)
        return

    oldres = old.calc.results
    newres = new.calc.results
    common_keys = set(oldres).intersection(newres)

    for key in common_keys:
        if np.any(oldres[key] != newres[key]):
            configs.append(new)
            return
    else:
        oldres.update(newres)


def read_gaussian_out(fd, index=-1):
    configs = []
    atoms = None
    energy = None
    dipole = None
    forces = None
    for line in fd:
        line = line.strip()
        if line.startswith(r'1\1\GINC'):
            # We've reached the "archive" block at the bottom, stop parsing
            break

        if (line == 'Input orientation:'
                or line == 'Z-Matrix orientation:'):
            if atoms is not None:
                atoms.calc = SinglePointCalculator(
                    atoms, energy=energy, dipole=dipole, forces=forces,
                )
                _compare_merge_configs(configs, atoms)
            atoms = None
            energy = None
            dipole = None
            forces = None

            numbers = []
            positions = []
            pbc = np.zeros(3, dtype=bool)
            cell = np.zeros((3, 3))
            npbc = 0
            # skip 4 irrelevant lines
            for _ in range(4):
                fd.readline()
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                number = int(match.group(1))
                pos = list(map(float, match.group(2, 3, 4)))
                if number == -2:
                    pbc[npbc] = True
                    cell[npbc] = pos
                    npbc += 1
                else:
                    numbers.append(max(number, 0))
                    positions.append(pos)
            atoms = Atoms(numbers, positions, pbc=pbc, cell=cell)
        elif (line.startswith('Energy=')
                or line.startswith('SCF Done:')):
            # Some semi-empirical methods (Huckel, MINDO3, etc.),
            # or SCF methods (HF, DFT, etc.)
            energy = float(line.split('=')[1].split()[0].replace('D', 'e'))
            energy *= Hartree
        elif (line.startswith('E2 =') or line.startswith('E3 =')
                or line.startswith('E4(') or line.startswith('DEMP5 =')
                or line.startswith('E2(')):
            # MP{2,3,4,5} energy
            # also some double hybrid calculations, like B2PLYP
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif line.startswith('Wavefunction amplitudes converged. E(Corr)'):
            # "correlated method" energy, e.g. CCSD
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif _re_l716.match(line):
            # Sometimes Gaussian will print "Rotating derivatives to
            # standard orientation" after the matched line (which looks like
            # "(Enter /opt/gaussian/g16/l716.exe)", though the exact path
            # depends on where Gaussian is installed). We *skip* the dipole
            # in this case, because it might be rotated relative to the input
            # orientation (and also it is numerically different even if the
            # standard orientation is the same as the input orientation).
            line = fd.readline().strip()
            if not line.startswith('Dipole'):
                continue
            dip = line.split('=')[1].replace('D', 'e')
            tokens = dip.split()
            dipole = []
            # dipole elements can run together, depending on what method was
            # used to calculate them. First see if there is a space between
            # values.
            if len(tokens) == 3:
                dipole = list(map(float, tokens))
            elif len(dip) % 3 == 0:
                # next, check if the number of tokens is divisible by 3
                nchars = len(dip) // 3
                for i in range(3):
                    dipole.append(float(dip[nchars * i:nchars * (i + 1)]))
            else:
                # otherwise, just give up on trying to parse it.
                dipole = None
                continue
            # this dipole moment is printed in atomic units, e-Bohr
            # ASE uses e-Angstrom for dipole moments.
            dipole = np.array(dipole) * Bohr
        elif _re_forceblock.match(line):
            # skip 2 irrelevant lines
            fd.readline()
            fd.readline()
            forces = []
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                forces.append(list(map(float, match.group(2, 3, 4))))
            forces = np.array(forces) * Hartree / Bohr
    if atoms is not None:
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, dipole=dipole, forces=forces,
        )
        _compare_merge_configs(configs, atoms)
    return configs[index]
