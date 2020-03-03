import pytest
from typing import Iterable

import numpy as np
from ase.dft.doscollection import (DOSCollection,
                                   GridDOSCollection,
                                   RawDOSCollection)
from ase.dft.dosdata import DOSData, RawDOSData, GridDOSData


class MinimalDOSCollection(DOSCollection):
    """Inherit from abstract base class to check its features"""
    def __init__(self, dos_series: Iterable[DOSData]) -> None:
        super().__init__(dos_series)


class YetAnotherDOSCollection(DOSCollection):
    """Inherit from abstract base class to check its features"""
    def __init__(self, dos_series: Iterable[DOSData]) -> None:
        super().__init__(dos_series)


class TestDOSCollection:
    @pytest.fixture
    def rawdos(self):
        return RawDOSData([1., 2., 4.], [2., 3., 2.],
                          info={'my_key': 'my_value'})

    @pytest.fixture
    def another_rawdos(self):
        return RawDOSData([3., 2., 5.], [1., 0., 2.],
                          info={'other_key': 'other_value'})

    @pytest.mark.parametrize('n_entries', [0, 1, 3])
    def test_sequence(self, rawdos, n_entries):
        dc = MinimalDOSCollection([rawdos] * n_entries)
        assert len(dc) == n_entries
        for i in range(n_entries):
            assert dc[i] == rawdos

        with pytest.raises(IndexError):
            dc[n_entries + 1]
        with pytest.raises(TypeError):
            dc['hello']

    def test_slicing(self, rawdos, another_rawdos):
        dc = MinimalDOSCollection([rawdos, another_rawdos, rawdos])
        assert dc[1:] == [another_rawdos, rawdos]
        assert dc[:-1] == [rawdos, another_rawdos]

    equality_data = [([], [], True),
                     ([rawdos], [rawdos], True),
                     ([rawdos, another_rawdos],
                      [rawdos, another_rawdos], True),
                     ([], [rawdos], False),
                     ([rawdos], [], False),
                     ([rawdos, another_rawdos], [rawdos], False),
                     ([rawdos, another_rawdos],
                      [another_rawdos, rawdos], False)]

    @pytest.mark.parametrize('series_1, series_2, isequal', equality_data)
    def test_collection_equality(self, rawdos, another_rawdos,
                                 series_1, series_2, isequal):
        assert (MinimalDOSCollection(series_1)
                == MinimalDOSCollection(series_2)) == isequal

    @pytest.mark.parametrize('other', [True, 1, 0.5, 'string', rawdos])
    def test_equality_wrongtype(self, rawdos, other):
        assert not MinimalDOSCollection([rawdos]) == other

    def test_addition(self, rawdos, another_rawdos):
        dc = MinimalDOSCollection([rawdos])

        double_dc = dc + dc
        assert len(double_dc) == 2
        assert double_dc[0] == rawdos
        assert double_dc[1] == rawdos

        assert (dc + MinimalDOSCollection([another_rawdos])
                == dc + another_rawdos)

        with pytest.raises(TypeError):
            MinimalDOSCollection([rawdos]) + YetAnotherDOSCollection([rawdos])

    
    @pytest.mark.parametrize('options', [{'x': [1., 1.1, 1.2],
                                          'width': 1.3,
                                          'smearing': 'Gauss'},
                                         {'x': [1.7, 2.1, 2.0],
                                          'width': 3.4,
                                          'smearing': 'Gauss'}])
    def test_sample(self, rawdos, another_rawdos, options):
        dc = MinimalDOSCollection([rawdos, another_rawdos])
        sampled_data = dc.sample(**options)
        for i, data in enumerate((rawdos, another_rawdos)):
            # Check consistency with individual DOSData objects
            assert np.allclose(sampled_data[i, :], data.sample(**options))
            # Check we aren't trivially comparing zeros
            assert np.all(sampled_data)

    sample_grid_options = [{'npts': 10, 'xmin': -2, 'xmax': 10,
                            'padding': 3, 'width': 1},
                           {'npts': 12, 'xmin': 0, 'xmax': 4,
                            'padding': 2.1, 'width': 2.3}]
    
    @pytest.mark.parametrize('options', sample_grid_options)
    def test_sample_grid(self, rawdos, another_rawdos, options):
        ref_min = min(rawdos.get_energies())
        ref_max = max(another_rawdos.get_energies())
        
        # Check auto minimum
        dc = MinimalDOSCollection([rawdos, another_rawdos])
        energies, dos = dc.sample_grid(10, xmax=options['xmax'],
                                       padding=options['padding'],
                                       width=options['width'])
        assert pytest.approx(energies[0]) == ref_min - options['padding'] * options['width']
        assert pytest.approx(energies[-1]) == options['xmax']

        # Check auto maximum
        energies, dos = dc.sample_grid(10, xmin=options['xmin'],
                                       padding=options['padding'],
                                       width=options['width'])
        assert pytest.approx(energies[0]) == options['xmin']
        assert (pytest.approx(energies[-1])
                == ref_max + options['padding'] * options['width'])

        # Check values
        energies, dos = dc.sample_grid(**options)
        for i, data in enumerate((rawdos, another_rawdos)):
            assert np.allclose(dos[i, :], data.sample_grid(**options)[1])


class TestRawDOSCollection:
    @pytest.fixture
    def griddos(self):
        energies = np.linspace(1, 10, 7)
        weights = np.sin(energies)
        return GridDOSData(energies, weights, info={'my_key': 'my_value'})

    def test_init(self, griddos):
        with pytest.raises(TypeError):
            RawDOSCollection([griddos])


class TestGridDOSCollection:
    @pytest.fixture
    def griddos(self):
        energies = np.linspace(1, 10, 7)
        weights = np.sin(energies)
        return GridDOSData(energies, weights, info={'my_key': 'my_value'})

    @pytest.fixture
    def another_griddos(self):
        energies = np.linspace(1, 10, 7)
        weights = np.cos(energies)
        return GridDOSData(energies, weights, info={'my_key': 'other_value'})

    def test_init_errors(self, griddos):
        with pytest.raises(TypeError):
            GridDOSCollection([RawDOSData([1.], [1.])])
        with pytest.raises(ValueError):
            energies = np.linspace(1, 10, 7) + 1
            GridDOSCollection([griddos,
                               GridDOSData(energies, np.sin(energies))])
        with pytest.raises(ValueError):
            energies = np.linspace(1, 10, 6)
            GridDOSCollection([griddos,
                               GridDOSData(energies, np.sin(energies))])

    def test_sequence(self, griddos, another_griddos):
        gdc = GridDOSCollection([griddos, another_griddos])

        for i, (coll_dosdata, dosdata) in enumerate(zip(gdc,
                                                        [griddos,
                                                         another_griddos])):
            assert coll_dosdata == dosdata
            assert gdc[i] == dosdata

    def test_slicing(self, griddos, another_griddos):
        gdc = GridDOSCollection([griddos, another_griddos, griddos])

        assert gdc[1:] == [another_griddos, griddos]
        assert gdc[:-1] == [griddos, another_griddos]

        with pytest.raises(TypeError):
            gdc['string']
