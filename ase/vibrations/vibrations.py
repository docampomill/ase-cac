"""Vibrational modes."""

import sys
from math import sin, pi, sqrt, log

import numpy as np

import ase.units as units
from ase.io.trajectory import Trajectory
from ase.parallel import world, paropen

from ase.utils.filecache import get_json_cache
from ase.calculators.singlepoint import SinglePointCalculator


class Vibrations:
    """Class for calculating vibrational modes using finite difference.

    The vibrational modes are calculated from a finite difference
    approximation of the Hessian matrix.

    The *summary()*, *get_energies()* and *get_frequencies()* methods all take
    an optional *method* keyword.  Use method='Frederiksen' to use the method
    described in:

      T. Frederiksen, M. Paulsson, M. Brandbyge, A. P. Jauho:
      "Inelastic transport theory from first-principles: methodology and
      applications for nanoscale devices", Phys. Rev. B 75, 205413 (2007)

    atoms: Atoms object
        The atoms to work on.
    indices: list of int
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.
    name: str
        Name to use for files.
    delta: float
        Magnitude of displacements.
    nfree: int
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.

    Example:

    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> from ase.optimize import BFGS
    >>> from ase.vibrations import Vibrations
    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)],
    ...            calculator=EMT())
    >>> BFGS(n2).run(fmax=0.01)
    BFGS:   0  16:01:21        0.440339       3.2518
    BFGS:   1  16:01:21        0.271928       0.8211
    BFGS:   2  16:01:21        0.263278       0.1994
    BFGS:   3  16:01:21        0.262777       0.0088
    >>> vib = Vibrations(n2)
    >>> vib.run()
    Writing vib.eq.json
    Writing vib.0x-.json
    Writing vib.0x+.json
    Writing vib.0y-.json
    Writing vib.0y+.json
    Writing vib.0z-.json
    Writing vib.0z+.json
    Writing vib.1x-.json
    Writing vib.1x+.json
    Writing vib.1y-.json
    Writing vib.1y+.json
    Writing vib.1z-.json
    Writing vib.1z+.json
    >>> vib.summary()
    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    2.5      20.4
    4    2.5      20.4
    5  152.6    1230.8
    ---------------------
    Zero-point energy: 0.079 eV
    >>> vib.write_mode(-1)  # write last mode to trajectory file

    """

    def __init__(self, atoms, indices=None, name='vib', delta=0.01, nfree=2):
        assert nfree in [2, 4]
        self.atoms = atoms
        self.calc = atoms.calc
        if indices is None:
            indices = range(len(atoms))
        self.indices = np.asarray(indices)
        #self.name = name
        self.delta = delta
        self.nfree = nfree
        self.H = None
        self.ir = None

        self.cache = get_json_cache(name)

    @property
    def name(self):
        return str(self.cache.directory)

    def run(self):
        """Run the vibration calculations.

        This will calculate the forces for 6 displacements per atom +/-x,
        +/-y, +/-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .json), which must be deleted before restarting the
        job. Otherwise the forces will not be calculated for that
        displacement.

        Note that the calculations for the different displacements can be done
        simultaneously by several independent processes. This feature relies
        on the existence of files and the subsequent creation of the file in
        case it is not found.

        If the program you want to use does not have a calculator in ASE, use
        ``iterdisplace`` to get all displaced structures and calculate the forces
        on your own.
        """

        if not self.cache.writable:
            raise RuntimeError(
                'Cannot run calculation.  '
                'Cache must be removed or split in order '
                'to have only one sort of data structure at a time.')

        for name, atoms in self.iterdisplace(inplace=True):
            with self.cache.lock(name) as handle:
                if handle is None:
                    continue

                result = self.calculate(atoms, handle)

                if world.rank == 0:
                    handle.save(result)

    def iterdisplace(self, inplace=False):
        """Yield name and atoms object for initial and displaced structures.

        Use this to export the structures for each single-point calculation
        to an external program instead of using ``run()``. Then save the
        calculated gradients to <name>.json and continue using this instance.
        """
        atoms = self.atoms if inplace else self.atoms.copy()
        yield self.name + '.eq', atoms
        for name, a, i, disp in self.displacements():
            if not inplace:
                atoms = self.atoms.copy()
            pos0 = atoms.positions[a, i]
            atoms.positions[a, i] += disp
            yield name, atoms

            if inplace:
                atoms.positions[a, i] = pos0

    def iterimages(self):
        """Yield initial and displaced structures."""
        for name, atoms in self.iterdisplace():
            yield atoms

    def displacements(self):
        for a in self.indices:
            for i in range(3):
                for sign in [-1, 1]:
                    for ndis in range(1, self.nfree // 2 + 1):
                        name = '%s.%d%s%s' % (self.name, a, 'xyz'[i],
                                              ndis * ' +-'[sign])
                        disp = ndis * sign * self.delta
                        yield name, a, i, disp

    def calculate(self, atoms, handle):
        results = {}
        results['forces'] = self.calc.get_forces(atoms)

        if self.ir:
            results['dipole'] = self.calc.get_dipole_moment(atoms)

        return results

    def clean(self, empty_files=False, combined=True):
        """Remove json-files.

        Use empty_files=True to remove only empty files and
        combined=False to not remove the combined file.

        """

        if world.rank != 0:
            return 0

        if empty_files:
            return self.cache.strip_empties()  # XXX Fails on combined cache

        nfiles = self.cache.filecount()
        self.cache.clear()
        return nfiles

    def combine(self):
        """Combine json-files to one file ending with '.all.json'.

        The other json-files will be removed in order to have only one sort
        of data structure at a time.

        """
        nelements_before = self.cache.filecount()
        self.cache = self.cache.combine()
        return nelements_before

    def split(self):
        """Split combined json-file.

        The combined json-file will be removed in order to have only one
        sort of data structure at a time.

        """
        count = self.cache.filecount()
        self.cache = self.cache.split()
        return count

    def read(self, method='standard', direction='central'):
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ['standard', 'frederiksen']
        assert self.direction in ['central', 'forward', 'backward']

        n = 3 * len(self.indices)
        H = np.empty((n, n))
        r = 0

        data = dict(self.cache)
        forces = {key: value['forces'] for key, value in data.items()}

        if direction != 'central':
            feq = forces['eq']

        for a in self.indices:
            for i in 'xyz':
                token = f'{self.name}.{a}{i}'
                fminus = forces[token + '-']
                fplus = forces[token + '+']
                if self.method == 'frederiksen':
                    fminus[a] -= fminus.sum(0)
                    fplus[a] -= fplus.sum(0)
                if self.nfree == 4:
                    fminusminus = forces[token + '--']
                    fplusplus = forces[token + '++']
                    if self.method == 'frederiksen':
                        fminusminus[a] -= fminusminus.sum(0)
                        fplusplus[a] -= fplusplus.sum(0)
                if self.direction == 'central':
                    if self.nfree == 2:
                        H[r] = .5 * (fminus - fplus)[self.indices].ravel()
                    else:
                        H[r] = H[r] = (-fminusminus +
                                       8 * fminus -
                                       8 * fplus +
                                       fplusplus)[self.indices].ravel() / 12.0
                elif self.direction == 'forward':
                    H[r] = (feq - fplus)[self.indices].ravel()
                else:
                    assert self.direction == 'backward'
                    H[r] = (fminus - feq)[self.indices].ravel()
                H[r] /= 2 * self.delta
                r += 1
        H += H.copy().T
        self.H = H
        m = self.atoms.get_masses()
        if 0 in [m[index] for index in self.indices]:
            raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

        self.im = np.repeat(m[self.indices]**-0.5, 3)
        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # Conversion factor:
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5

    def get_energies(self, method='standard', direction='central', **kw):
        """Get vibration energies in eV."""

        if (self.H is None or method.lower() != self.method or
            direction.lower() != self.direction):
            self.read(method, direction, **kw)
        return self.hnu

    def get_frequencies(self, method='standard', direction='central'):
        """Get vibration frequencies in cm^-1."""

        s = 1. / units.invcm
        return s * self.get_energies(method, direction)

    def summary(self, method='standard', direction='central', freq=None,
                log=sys.stdout):
        """Print a summary of the vibrational frequencies.

        Parameters:

        method : string
            Can be 'standard'(default) or 'Frederiksen'.
        direction: string
            Direction for finite differences. Can be one of 'central'
            (default), 'forward', 'backward'.
        freq : numpy array
            Optional. Can be used to create a summary on a set of known
            frequencies.
        log : if specified, write output to a different location than
            stdout. Can be an object with a write() method or the name of a
            file to create.
        """

        if isinstance(log, str):
            log = paropen(log, 'a')
        write = log.write

        s = 0.01 * units._e / units._c / units._hplanck
        if freq is not None:
            hnu = freq / s
        else:
            hnu = self.get_energies(method, direction)
        write('---------------------\n')
        write('  #    meV     cm^-1\n')
        write('---------------------\n')
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            write('%3d %6.1f%s  %7.1f%s\n' % (n, 1000 * e, c, s * e, c))
        write('---------------------\n')
        write('Zero-point energy: %.3f eV\n' %
              self.get_zero_point_energy(freq=freq))

    def get_zero_point_energy(self, freq=None):
        if freq is None:
            return 0.5 * self.hnu.real.sum()
        else:
            s = 0.01 * units._e / units._c / units._hplanck
            return 0.5 * freq.real.sum() / s

    def get_mode(self, n):
        """Get mode number ."""
        mode = np.zeros((len(self.atoms), 3))
        mode[self.indices] = (self.modes[n] * self.im).reshape((-1, 3))
        return mode

    def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.get_energies()):
                if abs(energy) > 1e-5:
                    self.write_mode(n=index, kT=kT, nimages=nimages)
            return
        mode = self.get_mode(n) * sqrt(kT / abs(self.hnu[n]))

        pos0 = self.atoms.get_positions()
        n %= 3 * len(self.indices)
        atoms = self.atoms.copy()

        with Trajectory('%s.%d.traj' % (self.name, n), 'w') as traj:
            for x in np.linspace(0, 2 * pi, nimages, endpoint=False):
                atoms.set_positions(pos0 + sin(x) * mode)
                traj.write(atoms)

    def show_as_force(self, n, scale=0.2):
        mode = self.get_mode(n) * len(self.hnu) * scale
        calc = SinglePointCalculator(self.atoms, forces=mode)
        self.atoms.calc = calc
        self.atoms.edit()

    def write_jmol(self):
        """Writes file for viewing of the modes with jmol."""

        with open(self.name + '.xyz', 'w') as fd:
            self._write_jmol(fd)

    def _write_jmol(self, fd):
        symbols = self.atoms.get_chemical_symbols()
        freq = self.get_frequencies()
        for n in range(3 * len(self.indices)):
            fd.write('%6d\n' % len(self.atoms))

            if freq[n].imag != 0:
                c = 'i'
                freq[n] = freq[n].imag
            else:
                c = ' '

            fd.write('Mode #%d, f = %.1f%s cm^-1' % (n, freq[n], c))
            if self.ir:
                fd.write(', I = %.4f (D/Å)^2 amu^-1.\n' % self.intensities[n])
            else:
                fd.write('.\n')

            mode = self.get_mode(n)
            for i, pos in enumerate(self.atoms.positions):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f \n' %
                         (symbols[i], pos[0], pos[1], pos[2],
                          mode[i, 0], mode[i, 1], mode[i, 2]))

    def fold(self, frequencies, intensities,
             start=800.0, end=4000.0, npts=None, width=4.0,
             type='Gaussian', normalize=False):
        """Fold frequencies and intensities within the given range
        and folding method (Gaussian/Lorentzian).
        The energy unit is cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """

        lctype = type.lower()
        assert lctype in ['gaussian', 'lorentzian']
        if not npts:
            npts = int((end - start) / width * 10 + 1)
        prefactor = 1
        if lctype == 'lorentzian':
            intensities = intensities * width * pi / 2.
            if normalize:
                prefactor = 2. / width / pi
        else:
            sigma = width / 2. / sqrt(2. * log(2.))
            if normalize:
                prefactor = 1. / sigma / sqrt(2 * pi)

        # Make array with spectrum data
        spectrum = np.empty(npts)
        energies = np.linspace(start, end, npts)
        for i, energy in enumerate(energies):
            energies[i] = energy
            if lctype == 'lorentzian':
                spectrum[i] = (intensities * 0.5 * width / pi /
                               ((frequencies - energy)**2 +
                                0.25 * width**2)).sum()
            else:
                spectrum[i] = (intensities *
                               np.exp(-(frequencies - energy)**2 /
                                      2. / sigma**2)).sum()
        return [energies, prefactor * spectrum]

    def write_dos(self, out='vib-dos.dat', start=800, end=4000,
                  npts=None, width=10,
                  type='Gaussian', method='standard', direction='central'):
        """Write out the vibrational density of states to file.

        First column is the wavenumber in cm^-1, the second column the
        folded vibrational density of states.
        Start and end points, and width of the Gaussian/Lorentzian
        should be given in cm^-1."""
        frequencies = self.get_frequencies(method, direction).real
        intensities = np.ones(len(frequencies))
        energies, spectrum = self.fold(frequencies, intensities,
                                       start, end, npts, width, type)

        # Write out spectrum in file.
        outdata = np.empty([len(energies), 2])
        outdata.T[0] = energies
        outdata.T[1] = spectrum

        with open(out, 'w') as fd:
            fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
            fd.write('# [cm^-1] arbitrary\n')
            for row in outdata:
                fd.write('%.3f  %15.5e\n' %
                         (row[0], row[1]))
