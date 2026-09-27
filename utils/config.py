"""Configuration values shared by the ZTE-STAR example notebooks.

The values in this module define the trajectory geometry, dead-time handling,
reconstruction settings, and timing used by the two public notebooks. Keeping
the settings in one place makes the examples easier to inspect and prevents
small parameter differences between trajectory generation and reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"


@dataclass
class Settings:
    """Container for acquisition and reconstruction parameters.

    Attributes:
        M: Number of spiral-phyllotaxis interleaves.
        P: Number of positive spokes per interleave. Paired acquisition doubles
            this count because each positive spoke is followed by its negative
            counterpart.
        k: Azimuthal skip factor used by the spiral-phyllotaxis ordering.
        NR: Number of acquired readout samples per half spoke.
        fov: Relative field-of-view scaling along the three spatial axes.
        n_adc_delay: Readout delay expressed in sample units.
        radial_matrix_size: Matrix size used to construct the conventional
            radial reference trajectory.
        radial_AF: Acceleration factor used for the conventional radial
            trajectory.
        radial_n_pro: Number of conventional radial spokes per repetition.
        radial_n_rep: Number of repeated conventional radial frames in the
            included simulated dataset.
        n_gap: Number of central paired-spoke samples restored by algebraic
            dead-time infilling.
        over_sampling: Readout oversampling factor assumed by the algebraic
            infilling operator.
        matrix_size: Three-dimensional reconstruction matrix size.
        L: Convenience alias for the cubic matrix length used by the notebooks.
        n_window: Number of spiral-phyllotaxis interleaves grouped into one TTV
            reconstruction frame.
        lamda: Temporal total-variation regularization weight.
        n_iter: Maximum number of reconstruction iterations.
        n_calc_seg: Number of temporal calculation segments.
        finite_support: Whether to apply a spherical finite-support mask.
        radial_n_window: Window label used when saving conventional radial
            gridding results.
        TR_sec: Repetition time in seconds.
    """

    # Spiral-phyllotaxis acquisition parameters.
    M: int = 144
    P: int = 144
    k: int = 55
    NR: int = 128
    fov: tuple = (1, 1, 1)
    n_adc_delay: float = 1.5

    # Conventional radial reference parameters.
    radial_matrix_size: int = 64
    radial_AF: float = 4.5
    radial_n_pro: int = 2948
    radial_n_rep: int = 16

    # Paired-spoke dead-time infilling parameters.
    n_gap: int = 5
    over_sampling: int = 4

    # Reconstruction parameters.
    matrix_size: tuple = (64, 64, 64)
    L: int = 64
    n_window: int = 3
    lamda: float = 1e-6
    n_iter: int = 50
    n_calc_seg: int = 1
    finite_support: bool = True
    radial_n_window: int = 5

    # Acquisition timing used to convert spoke counts to seconds.
    TR_sec: float = 0.0012

    @property
    def n_spokes(self) -> int:
        """Return the total number of paired half spokes in one acquisition."""
        return 2 * self.M * self.P

    @property
    def spokes_per_frame(self) -> int:
        """Return the number of half spokes grouped into one TTV frame."""
        return 2 * self.P * self.n_window

    @property
    def frame_sec(self) -> float:
        """Return the nominal duration of one TTV frame in seconds."""
        return self.spokes_per_frame * self.TR_sec


PAPER = Settings()


def get() -> Settings:
    """Return the shared settings instance used by the example notebooks.

    Returns:
        The module-level :class:`Settings` instance.
    """
    return PAPER
