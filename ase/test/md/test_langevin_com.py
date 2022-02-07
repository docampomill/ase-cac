
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary
from ase.md.langevin import Langevin
from numpy.linalg import norm

def test_langevin_com():
    # parameters
    size = 2
    T = 300
    dt = 0.01

    # setup
    #atoms = bulk('Al', 'fcc', a=4.0).repeat(size)
    atoms = bulk('CuAg', 'rocksalt', a=4.0).repeat(size)
    atoms.pbc = False
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)
    dyn = Langevin(atoms, dt * units.fs, temperature_K=T, friction=0.02)

    # run NVT
    mtot = atoms.get_momenta().sum(axis=0)

    print('initial momenta', mtot)
    print('initial forces', atoms.get_forces().sum(axis=0))

    dyn.run(10)

    m2 = atoms.get_momenta().sum(axis=0)
    print('momenta', m2)
    print('forces', atoms.get_forces().sum(axis=0))
    print()

    assert norm(m2) < 1e-8
