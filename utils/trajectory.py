"""Trajectory generation for ZTE-STAR and conventional radial sampling.

This module contains only the trajectory operations required by the public
example notebooks. Trajectories are represented as NumPy arrays with shape
`(3, n_readout, n_spokes)`, where the first axis stores kx, ky, and kz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _radial_angles(n, factor):
    """Return the integer number of radial angles for a sampling factor.

    Args:
        n: Reference matrix dimension.
        factor: Relative angular sampling factor.

    Returns:
        Integer number of angular samples.
    """
    return np.ceil((np.pi * n * factor) / 2).astype(int)


def _centered_angle(i, n):
    """Return the center angle of one of `n` equal angular intervals.

    Args:
        i: Zero-based interval index.
        n: Number of intervals.

    Returns:
        Center angle in radians.
    """
    return np.pi * (i + 0.5) / n


def calc_n_pro(matrix_size, under_sampling):
    """Calculate the number of half-sphere radial projections.

    Args:
        matrix_size: Isotropic reconstruction matrix dimension.
        under_sampling: Radial undersampling factor.

    Returns:
        Number of projection directions before adding opposite spokes.
    """
    usamp = np.sqrt(under_sampling)
    n_theta = _radial_angles(matrix_size, 1 / usamp)
    n_pro = 0
    for i_theta in range(n_theta):
        theta = _centered_angle(i_theta, n_theta)
        n_phi = _radial_angles(matrix_size, np.sin(theta) / usamp)
        n_pro += n_phi
    return n_pro


def reorder_projections_silent(n_theta, nphi_list, g):
    """Reverse alternating polar rings to produce a continuous traversal.

    The reordering changes only acquisition order. It does not alter the set of
    radial directions.

    Args:
        n_theta: Number of polar rings.
        nphi_list: Number of azimuthal samples in each polar ring.
        g: Projection direction array with shape `(n_projections, 3)`.

    Returns:
        The reordered projection array. The input array is modified in place
        and returned for convenience.
    """
    i = 0
    for i_theta in range(n_theta):
        nphi = nphi_list[i_theta]
        if i_theta % 2 == 1 and nphi > 1:
            g[i:i + nphi] = g[i:i + nphi][::-1]
        i += nphi
    return g


def generate_bruker_radial_dirs(matrix_size, AF=1.0, reorder=True):
    """Generate conventional radial projection directions.

    Directions are first generated over a half sphere and are then duplicated
    with opposite signs so that the final set contains paired +k/-k spokes.

    Args:
        matrix_size: Isotropic matrix dimension used to determine angular
            sampling density.
        AF: Radial acceleration factor.
        reorder: Whether to reverse alternating polar rings to obtain the
            conventional continuous ordering.

    Returns:
        A DataFrame with columns `x`, `y`, and `z`, containing unit
        direction vectors for all paired spokes.
    """
    n_pro_all = calc_n_pro(matrix_size, 1.0)
    s = np.sqrt(AF)
    n_theta = _radial_angles(matrix_size, 1.0 / s)
    projs = []
    radial_n_phi = []

    for theta_index in range(n_theta):
        theta = _centered_angle(theta_index, n_theta)
        n_phi = _radial_angles(matrix_size, np.sin(theta) / s)
        radial_n_phi.append(n_phi)

        for phi_index in range(n_phi):
            phi = _centered_angle(phi_index, n_phi)
            projs.append(
                (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                )
            )

    directions = np.asarray(projs, dtype=float)
    if reorder:
        directions = reorder_projections_silent(
            n_theta, radial_n_phi, directions
        )

    n_pro = len(directions)
    directions = np.concatenate([directions, -directions], axis=0)
    print(
        "N_theta =",
        n_theta,
        "N_proj =",
        len(directions),
        "actual_AF =",
        n_pro_all / n_pro,
    )
    return pd.DataFrame(directions, columns=["x", "y", "z"])


def calc_spiral_phyllotaxis_dirs(M, P, k, z_factor=1.0):
    """Calculate base spiral-phyllotaxis direction vectors.

    The polar coordinate is distributed uniformly in z, while the azimuth is
    advanced using the golden-angle relation and the skip factor `k`.

    Args:
        M: Number of interleaves.
        P: Number of base spokes per interleave.
        k: Azimuthal skip factor.
        z_factor: Polar-range scaling. A value of 1.0 covers the full sphere
            before explicit pairing; 0.5 is retained for the half-range mode.

    Returns:
        NumPy array with shape `(M * P, 3)` containing unit direction
        vectors.

    Raises:
        ValueError: If `z_factor` is not 1.0 or 0.5.
    """
    n_total = M * P

    if z_factor == 1.0:
        phi_g = np.pi * (3.0 - np.sqrt(5.0))
        dz = (2.0 * z_factor) / (n_total - 1)
    elif z_factor == 0.5:
        phi_g = np.pi * (89.0 / 144.0)
        dz = (2.0 * z_factor) / (n_total - 1)
    else:
        raise ValueError("z_factor must be 1.0 or 0.5")

    # Build the interleaf and within-interleaf indices as broadcastable arrays.
    i = np.arange(P)[:, None]
    j = np.arange(M)[None, :]

    # Advance the azimuth according to the selected phyllotaxis skip factor.
    phi = ((i * k + j) * phi_g) % (2.0 * np.pi)

    # Uniform z spacing provides approximately uniform area sampling on a
    # sphere.
    z = 1.0 - (i * M + j) * dz
    theta = np.arccos(np.clip(z, -1.0, 1.0))

    sin_theta = np.sin(theta)
    x = (sin_theta * np.cos(phi)).ravel(order="C")
    y = (sin_theta * np.sin(phi)).ravel(order="C")
    z = z.ravel(order="C")
    return np.vstack([x, y, z]).T


def generate_spiral_phyllotaxis_dirs(
    M,
    P,
    k,
    half_spoke=True,
    paired=True,
):
    """Generate spiral-phyllotaxis directions in acquisition order.

    Args:
        M: Number of interleaves.
        P: Number of positive spokes per interleave.
        k: Azimuthal skip factor.
        half_spoke: Whether each readout represents a center-out half spoke.
        paired: Whether to append the negative counterpart of each interleaf.

    Returns:
        A DataFrame with columns `x`, `y`, and `z`.
    """
    z_factor = 1.0 if half_spoke else 0.5
    directions = calc_spiral_phyllotaxis_dirs(
        M, P, k, z_factor=z_factor
    )

    if paired:
        directions = np.concatenate(
            [
                np.concatenate(
                    [directions[j::M], -directions[j::M]], axis=0
                )
                for j in range(M)
            ],
            axis=0,
        )
    else:
        directions = np.concatenate(
            [directions[j::M] for j in range(M)], axis=0
        )

    return pd.DataFrame(directions, columns=list("xyz"))


def proj2traj(
    df,
    NR,
    norm=1,
    fov=(1, 1, 1),
    n_adc_delay=1.5,
    offset=None,
):
    """Expand direction vectors into sampled center-out trajectories.

    Args:
        df: DataFrame of unit projection directions with columns `x`, `y`,
            and `z`.
        NR: Number of acquired readout samples.
        norm: Global trajectory scaling factor.
        fov: Relative field-of-view scaling along x, y, and z.
        n_adc_delay: Readout delay in sample units. The integer part adds
            missing pre-readout points, while the fractional part shifts the
            first sampled position.
        offset: Optional explicit fractional sample offset. If omitted, the
            fractional part of `n_adc_delay` is used.

    Returns:
        A DataFrame containing shot indices and sampled x, y, and z
        coordinates for every readout point.
    """
    n_shot = len(df)

    kmax = np.array([1 / value for value in fov])
    kmax = norm * kmax / kmax.max()

    if n_adc_delay > 0:
        n_miss_point = int(n_adc_delay)
        if offset is None:
            offset = n_adc_delay % 1
    else:
        n_miss_point, offset = 0, 0

    n_points = NR + n_miss_point
    grad_x = np.zeros(n_points * n_shot)
    grad_y = np.zeros(n_points * n_shot)
    grad_z = np.zeros(n_points * n_shot)
    shot_ids = np.repeat(df.index.values, n_points)

    sample_positions = np.array(
        [sample + offset for sample in range(n_points)]
    )
    sample_positions = sample_positions / sample_positions.max()

    for shot in range(n_shot):
        dir_x = df["x"][shot]
        dir_y = df["y"][shot]
        dir_z = df["z"][shot]
        shot_slice = slice(shot * n_points, (shot + 1) * n_points)

        grad_x[shot_slice] = kmax[0] * dir_x * sample_positions
        grad_y[shot_slice] = kmax[1] * dir_y * sample_positions
        grad_z[shot_slice] = kmax[2] * dir_z * sample_positions

    return pd.DataFrame(
        {"shot": shot_ids, "x": grad_x, "y": grad_y, "z": grad_z}
    )


def traj2np(df, norm=1, pairspoke=False, n_seg=1):
    """Convert a trajectory DataFrame to a dense NumPy array.

    Args:
        df: Trajectory DataFrame produced by :func:`proj2traj`.
        norm: Additional trajectory scaling factor.
        pairspoke: Whether to concatenate each negative/positive spoke pair
            into one full-diameter readout.
        n_seg: Number of acquisition segments used to locate paired spokes.

    Returns:
        Dense trajectory array. Before the caller transposes axes, the array is
        indexed as `(spoke, readout, xyz)`.
    """
    n_shot = df["shot"].max() + 1
    stack = []

    if not pairspoke:
        for shot in range(n_shot):
            traj = df[df["shot"] == shot]
            traj = np.stack(
                [
                    traj["x"].to_numpy(),
                    traj["y"].to_numpy(),
                    traj["z"].to_numpy(),
                ],
                axis=-1,
            )
            stack.append(traj * norm)
    else:
        n_seg_one = n_shot // n_seg

        for segment in range(n_seg):
            for pair_index in range(n_seg_one // 2):
                pos_idx = (
                    (2 * segment) * (n_seg_one // 2) + pair_index
                )
                neg_idx = (
                    (2 * segment + 1) * (n_seg_one // 2) + pair_index
                )

                pos = df[df["shot"] == pos_idx]
                neg = df[df["shot"] == neg_idx]

                pos = np.stack(
                    [
                        pos["x"].to_numpy(),
                        pos["y"].to_numpy(),
                        pos["z"].to_numpy(),
                    ],
                    axis=-1,
                )
                neg = np.stack(
                    [
                        neg["x"].to_numpy(),
                        neg["y"].to_numpy(),
                        neg["z"].to_numpy(),
                    ],
                    axis=-1,
                )

                # Reverse the negative half before joining it to the positive
                # half so that samples progress continuously across k-space.
                stack.append(
                    np.concatenate((neg[::-1], pos), axis=0) * norm
                )

    return np.stack(stack, axis=0)


def _maybe_cast(array, dtype):
    """Cast an array only when a target dtype is explicitly requested."""
    return array if dtype is None else array.astype(dtype, copy=False)


def spiral_phyllotaxis(
    M: int = 144,
    P: int = 144,
    k: int = 55,
    NR: int = 128,
    fov=(1, 1, 1),
    n_adc_delay: float = 1.5,
    *,
    half_spoke: bool = True,
    paired: bool = True,
    pairspoke: bool = False,
    n_seg: int | None = None,
    dtype=None,
):
    """Build the sampled ZTE-STAR spiral-phyllotaxis trajectory.

    Args:
        M: Number of interleaves.
        P: Number of positive spokes per interleave.
        k: Azimuthal skip factor.
        NR: Number of acquired readout samples per half spoke.
        fov: Relative field-of-view scaling along x, y, and z.
        n_adc_delay: Readout delay in sample units.
        half_spoke: Whether to use center-out half-spoke readouts.
        paired: Whether to include the negative counterpart of each spoke.
        pairspoke: Whether to return joined +k/-k full-diameter trajectories.
        n_seg: Number of pairing segments. If omitted, paired-spoke output uses
            `M` segments and acquisition-order output uses one segment.
        dtype: Optional NumPy dtype for the returned array.

    Returns:
        Trajectory array with shape `(3, n_readout, n_spokes)`.
    """
    directions = generate_spiral_phyllotaxis_dirs(
        M=M,
        P=P,
        k=k,
        half_spoke=half_spoke,
        paired=paired,
    )
    traj_df = proj2traj(
        directions,
        NR=NR,
        fov=list(fov),
        n_adc_delay=n_adc_delay,
    )

    if n_seg is None:
        n_seg = M if pairspoke else 1

    traj = traj2np(
        traj_df,
        pairspoke=pairspoke,
        n_seg=n_seg,
    ).T
    return _maybe_cast(traj, dtype)


def bruker_radial(
    matrix_size: int = 64,
    AF: float = 4.5,
    NR: int = 128,
    fov=(1, 1, 1),
    n_adc_delay: float = 1.5,
    *,
    reorder: bool = True,
    pairspoke: bool = False,
    n_seg: int = 1,
    n_rep: int = 1,
    dtype=None,
):
    """Build the sampled conventional radial trajectory.

    Args:
        matrix_size: Isotropic matrix dimension used for angular sampling.
        AF: Radial acceleration factor.
        NR: Number of acquired readout samples per half spoke.
        fov: Relative field-of-view scaling along x, y, and z.
        n_adc_delay: Readout delay in sample units.
        reorder: Whether to use the conventional continuous ring ordering.
        pairspoke: Whether to return joined +k/-k full-diameter trajectories.
        n_seg: Number of pairing segments.
        n_rep: Number of repetitions concatenated in acquisition order.
        dtype: Optional NumPy dtype for the returned array.

    Returns:
        Trajectory array with shape `(3, n_readout, n_spokes)`.
    """
    directions = generate_bruker_radial_dirs(
        matrix_size=matrix_size,
        AF=AF,
        reorder=reorder,
    )
    traj_df = proj2traj(
        directions,
        NR=NR,
        fov=list(fov),
        n_adc_delay=n_adc_delay,
    )
    traj = traj2np(
        traj_df,
        pairspoke=pairspoke,
        n_seg=n_seg,
    ).T

    if n_rep != 1:
        traj = np.concatenate([traj] * int(n_rep), axis=-1)

    return _maybe_cast(traj, dtype)


def from_settings(settings, kind: str, *, pairspoke: bool, dtype=None):
    """Build a trajectory directly from a shared settings object.

    Args:
        settings: Settings instance containing trajectory parameters.
        kind: Either `"spiral"` or `"radial"`.
        pairspoke: Whether to return joined +k/-k trajectories.
        dtype: Optional NumPy dtype for the returned array.

    Returns:
        Trajectory array for the requested acquisition type.

    Raises:
        ValueError: If `kind` is neither `"spiral"` nor `"radial"`.
    """
    if kind == "spiral":
        return spiral_phyllotaxis(
            M=settings.M,
            P=settings.P,
            k=settings.k,
            NR=settings.NR,
            fov=settings.fov,
            n_adc_delay=settings.n_adc_delay,
            pairspoke=pairspoke,
            n_seg=settings.M if pairspoke else 1,
            dtype=dtype,
        )

    if kind == "radial":
        return bruker_radial(
            matrix_size=settings.radial_matrix_size,
            AF=settings.radial_AF,
            NR=settings.NR,
            fov=settings.fov,
            n_adc_delay=settings.n_adc_delay,
            pairspoke=pairspoke,
            n_seg=1,
            n_rep=1 if pairspoke else settings.radial_n_rep,
            dtype=dtype,
        )

    raise ValueError(f"kind must be 'spiral' or 'radial', got {kind!r}")
