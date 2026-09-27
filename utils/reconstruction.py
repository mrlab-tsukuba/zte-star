"""Dead-time infilling and image reconstruction for the ZTE-STAR examples.

The public notebook API consists of paired-spoke preprocessing, temporal
total-variation reconstruction for ZTE-STAR, and density-compensated gridding
for the conventional radial comparison. Internal helper functions are kept in
this module so the complete reconstruction path is visible in one place.
"""

from pathlib import Path
import os
import time

import numpy as np
import sigpy as sp
from sigpy.app import LinearLeastSquares
from tqdm import tqdm

from . import _linop as linop


def sphere_mask(shape=(64, 64, 64), diameter=None, radius=None, center=None):
    """Create a boolean spherical finite-support mask.

    Args:
        shape: Three-dimensional output shape in `(z, y, x)` order.
        diameter: Sphere diameter in voxel units. Used only when `radius` is
            not provided. If both diameter and radius are omitted, the smallest
            output dimension is used as the diameter.
        radius: Sphere radius in voxel units. If provided, this value takes
            precedence over `diameter`.
        center: Sphere center in voxel coordinates. If omitted, the center is
            placed at `(n - 1) / 2` along each axis so that even-sized arrays
            are centered between the two middle voxels.

    Returns:
        Boolean NumPy array with `True` inside or on the sphere and `False`
        outside it.
    """
    shape = np.asarray(shape, dtype=float)

    if center is None:
        # Place the geometric center between the two central voxels for even
        # matrix dimensions, for example at 31.5 for a size of 64.
        center = (shape - 1) / 2.0
    else:
        center = np.asarray(center, dtype=float)

    if radius is None:
        if diameter is None:
            diameter = float(shape.min())
        radius = float(diameter) / 2.0
    else:
        radius = float(radius)

    z, y, x = np.ogrid[
        :int(shape[0]),
        :int(shape[1]),
        :int(shape[2]),
    ]
    dist2 = (
        (z - center[0]) ** 2
        + (y - center[1]) ** 2
        + (x - center[2]) ** 2
    )
    return dist2 <= radius**2


def grouping(data, traj, n_window=8, M=144, P=144, n_segment=1):
    """Group continuous paired-spoke data into temporal reconstruction frames.

    Args:
        data: K-space array indexed as `(coil, readout, spoke)`.
        traj: Matching trajectory array indexed as `(3, readout, spoke)`.
        n_window: Number of interleaves assigned to one reconstruction frame.
        M: Number of interleaves in one complete spiral-phyllotaxis cycle.
        P: Number of paired-spoke units contributed by one interleave after
            preprocessing.
        n_segment: Number of concatenated acquisition segments.

    Returns:
        A tuple `(multiframe_data, multiframe_traj)`. The first array is
        indexed as `(frame, coil, readout, spoke)`, and the second as
        `(frame, readout, spoke, xyz)`.
    """
    multiframe_data = []
    multiframe_traj = []

    for frame in tqdm(range((M * n_segment) // n_window)):
        start = frame * n_window * P
        stop = (frame + 1) * n_window * P

        data_t = data[:, :, start:stop]
        traj_t = traj[..., start:stop]

        multiframe_data.append(data_t)
        multiframe_traj.append(
            np.stack(
                [traj_t[0], traj_t[1], traj_t[2]],
                axis=-1,
            )
        )

    multiframe_data = np.stack(multiframe_data, axis=0)
    multiframe_traj = np.stack(multiframe_traj, axis=0)
    return multiframe_data, multiframe_traj


def grouping_temp(data, traj, n_segment=1):
    """Convert pre-separated temporal segments to frame-first arrays.

    Args:
        data: K-space array whose final axis indexes temporal segments.
        traj: Matching trajectory array whose final axis indexes temporal
            segments.
        n_segment: Number of temporal segments to extract.

    Returns:
        A tuple `(multiframe_data, multiframe_traj)` with time on the leading
        axis.
    """
    multiframe_data = []
    multiframe_traj = []

    for segment in tqdm(range(n_segment)):
        data_t = data[..., segment]
        traj_t = traj[..., segment]

        multiframe_data.append(data_t)
        multiframe_traj.append(
            np.stack(
                [traj_t[0], traj_t[1], traj_t[2]],
                axis=-1,
            )
        )

    multiframe_data = np.stack(multiframe_data, axis=0)
    multiframe_traj = np.stack(multiframe_traj, axis=0)
    return multiframe_data, multiframe_traj


def recon_ttv(ksp, mps, traj, dcf, lamda=0.001, max_iter=10):
    """Reconstruct a dynamic series with L1 temporal total variation.

    Args:
        ksp: K-space data with shape `(T, C, P)`.
        mps: Coil sensitivity maps with shape `(C, L, L, L)`.
        traj: Non-Cartesian trajectory with shape `(T, P, 3)`.
        dcf: Density-compensation weights with shape `(T, 1, P)`.
        lamda: L1 temporal total-variation regularization weight.
        max_iter: Maximum number of ADMM iterations.

    Returns:
        A tuple `(recon, initial, runtime)`, where `recon` is the TTV
        reconstruction, `initial` is the density-compensated adjoint
        reconstruction, and `runtime` is the solver time in seconds.
    """
    n_frames, n_coils, n_points = ksp.shape
    n_frames, n_points, _ = traj.shape
    n_coils, matrix_size, _, _ = mps.shape

    device = sp.Device(0)
    ksp = sp.to_device(ksp, device)
    mps = sp.to_device(mps, device)
    traj = sp.to_device(traj, device)
    dcf = sp.to_device(dcf, device)

    reshape = linop.Reshape(
        (n_frames, 1, matrix_size, matrix_size, matrix_size),
        (n_frames, matrix_size, matrix_size, matrix_size),
    )
    multiply = linop.Multiply(reshape.oshape, mps)
    sensitivity = multiply * reshape
    nufft = linop.HDNUFFT(sensitivity.oshape, traj)
    weighting = linop.Multiply(nufft.oshape, dcf**0.5)

    # Compose the weighted non-Cartesian forward model A = WFS.
    forward = weighting * nufft * sensitivity

    # Penalize frame-to-frame finite differences along the temporal axis.
    temporal_difference = linop.FiniteDifference(forward.ishape, axes=[0])
    prox = sp.prox.L1Reg(temporal_difference.oshape, lamda=lamda)

    start_time = time.time()
    recon = LinearLeastSquares(
        A=forward,
        y=weighting * ksp,
        G=temporal_difference,
        proxg=prox,
        solver="ADMM",
        rho=0.1,
        max_iter=max_iter,
        show_pbar=False,
    ).run()
    runtime = time.time() - start_time

    recon = sp.to_device(recon, device=sp.Device(-1))
    initial = (forward.H * weighting * ksp).get()
    return recon, initial, runtime


def recon_gridding(ksp, mps, traj, dcf):
    """Reconstruct a dynamic series by density-compensated adjoint gridding.

    Args:
        ksp: K-space data with shape `(T, C, P)`.
        mps: Coil sensitivity maps with shape `(C, L, L, L)`.
        traj: Non-Cartesian trajectory with shape `(T, P, 3)`.
        dcf: Density-compensation weights with shape `(T, 1, P)`.

    Returns:
        A tuple `(recon, runtime)`, where `recon` is the
        density-compensated adjoint reconstruction and `runtime` is the
        reconstruction time in seconds.
    """
    n_frames, n_coils, n_points = ksp.shape
    n_frames, n_points, _ = traj.shape
    n_coils, matrix_size, _, _ = mps.shape

    device = sp.Device(0)
    ksp = sp.to_device(ksp, device)
    mps = sp.to_device(mps, device)
    traj = sp.to_device(traj, device)
    dcf = sp.to_device(dcf, device)

    reshape = linop.Reshape(
        (n_frames, 1, matrix_size, matrix_size, matrix_size),
        (n_frames, matrix_size, matrix_size, matrix_size),
    )
    multiply = linop.Multiply(reshape.oshape, mps)
    sensitivity = multiply * reshape
    nufft = linop.HDNUFFT(sensitivity.oshape, traj)
    weighting = linop.Multiply(nufft.oshape, dcf**0.5)

    # Use the same weighted forward model as the TTV path, but apply only its
    # adjoint so that the result is a conventional gridding reconstruction.
    forward = weighting * nufft * sensitivity

    start_time = time.time()
    recon = (forward.H * weighting * ksp).get()
    runtime = time.time() - start_time
    return recon, runtime


def traj_to_flat(trajectory, norm=32):
    """Flatten readout and spoke axes for SigPy's non-Cartesian operators.

    Args:
        trajectory: Array with shape `(T, Nr, Ns, ndim)`.
        norm: Coordinate scaling applied before flattening.

    Returns:
        Array with shape `(T, Nr * Ns, ndim)`.
    """
    norm = norm * (1 - 1e-4)
    n_frames, n_readout, n_spokes, ndim = trajectory.shape
    flat_traj = trajectory * norm
    return flat_traj.reshape(
        (n_frames, n_readout * n_spokes, ndim),
        order="C",
    )


def data_to_flat(data):
    """Flatten readout and spoke axes of frame-grouped k-space data.

    Args:
        data: Array with shape `(T, C, Nr, Ns)`.

    Returns:
        Array with shape `(T, C, Nr * Ns)`.
    """
    n_frames, n_coils, n_readout, n_spokes = data.shape
    return data.reshape(
        (n_frames, n_coils, n_readout * n_spokes),
        order="C",
    )


def report_finiteness(name, arr):
    """Print basic numerical sanity checks for an intermediate array.

    Args:
        name: Label printed before the diagnostic values.
        arr: Array-like object to inspect.
    """
    arr = np.asanyarray(arr)
    print(
        f"{name}: shape={arr.shape}, finite={np.isfinite(arr).all()}",
        f"nan={np.isnan(arr).any()}",
        f"inf={np.isinf(arr).any()}",
        f"max={np.max(arr):.3e}",
        f"min={np.min(arr):.3e}",
    )


def run_recon(
    data,
    traj,
    n_calc_seg=1,
    n_window=8,
    M=144,
    P=144,
    lamda=1e-5,
    matrix_size=(64, 64, 64),
    n_iter=10,
    prefix=None,
    save_root="./",
    finite_support=False,
):
    """Run segmented temporal-TV reconstruction and save intermediate results.

    Args:
        data: Prepared paired-spoke k-space data.
        traj: Prepared paired-spoke trajectory.
        n_calc_seg: Number of overlapping temporal calculation segments.
        n_window: Number of interleaves grouped into one reconstructed frame.
        M: Number of interleaves in one complete acquisition cycle.
        P: Number of paired-spoke units per interleave after preprocessing.
        lamda: Temporal total-variation regularization weight.
        matrix_size: Three-dimensional reconstruction matrix size.
        n_iter: Maximum number of ADMM iterations.
        prefix: Optional filename prefix for reconstruction outputs.
        save_root: Directory in which results and caches are written.
        finite_support: Whether to apply a spherical finite-support mask.
    """
    matrix_length, _, _ = matrix_size
    prefix = prefix + "_" if prefix is not None else ""

    # Concatenate repeated acquisition segments before retrospective temporal
    # grouping so that frame boundaries follow the continuous spoke stream.
    print("splitting data and trajectory into multiframe...")
    n_segment = data.shape[-1]
    data = np.concatenate(
        [data[..., segment] for segment in range(n_segment)],
        axis=-1,
    )
    traj = np.concatenate(n_segment * [traj], axis=-1)

    multiframe_data, multiframe_traj = grouping(
        data,
        traj,
        n_window=n_window,
        M=M,
        P=P,
        n_segment=n_segment,
    )
    n_frames_total = multiframe_data.shape[0]

    frames_per_segment = n_frames_total // n_calc_seg
    pad_size = frames_per_segment // 4

    cache_dir = os.path.join(
        save_root,
        f"w{n_window}_calcseg{n_calc_seg}cache",
    )
    os.makedirs(cache_dir, exist_ok=True)

    for section in range(n_calc_seg):
        # Add temporal overlap between calculation segments so that the TTV
        # penalty has neighboring frames near internal segment boundaries.
        if n_calc_seg == 1:
            left_pad = 0
            right_pad = 0
            section_data = multiframe_data
            section_traj = multiframe_traj
        elif section == n_calc_seg - 1:
            left_pad = pad_size
            right_pad = 0
            section_data = multiframe_data[
                frames_per_segment * section - left_pad:
            ]
            section_traj = multiframe_traj[
                frames_per_segment * section - left_pad:
            ]
        elif section == 0:
            left_pad = 0
            right_pad = pad_size
            section_data = multiframe_data[
                :frames_per_segment * (section + 1) + right_pad
            ]
            section_traj = multiframe_traj[
                :frames_per_segment * (section + 1) + right_pad
            ]
        else:
            left_pad = pad_size
            right_pad = pad_size
            section_data = multiframe_data[
                frames_per_segment * section - left_pad:
                frames_per_segment * (section + 1) + right_pad
            ]
            section_traj = multiframe_traj[
                frames_per_segment * section - left_pad:
                frames_per_segment * (section + 1) + right_pad
            ]

        n_frames = section_data.shape[0]
        n_coils = section_data.shape[1]

        section_traj = traj_to_flat(section_traj, matrix_length / 2)
        section_data = data_to_flat(section_data)

        print(
            f"calculation segment {section + 1} / {n_calc_seg}, "
            f"n_frames = {n_frames}, left_pad = {left_pad}, "
            f"right_pad = {right_pad}"
        )

        # Calculate Pipe-Menon density-compensation weights once per segment
        # and reuse the saved values on subsequent runs.
        print("calculating density compensation filter...")
        from sigpy.mri.dcf import pipe_menon_dcf

        report_finiteness("ksp", section_data)
        report_finiteness("traj", section_traj)

        dcf_path = os.path.join(cache_dir, f"dcf_sec{section}.npy")
        if not os.path.exists(dcf_path):
            dcf = pipe_menon_dcf(
                section_traj,
                img_shape=(
                    matrix_length,
                    matrix_length,
                    matrix_length,
                ),
                device=sp.Device(0),
            ).get()
            dcf = dcf.reshape(n_frames, 1, -1, order="C")
            dcf = dcf / np.max(abs(dcf))
            np.save(dcf_path, dcf)

        dcf = np.load(dcf_path)

        # Reconstruct the current temporal segment with the L1-TTV model.
        print("starting L1-TTV reconstruction...")
        mps = np.ones(
            (
                n_coils,
                matrix_length,
                matrix_length,
                matrix_length,
            ),
            dtype=np.complex64,
        ) / n_coils

        recon, initial, runtime = recon_ttv(
            section_data,
            mps,
            section_traj,
            dcf,
            lamda=lamda,
            max_iter=n_iter,
        )
        print(f"runtime: {runtime:.5f} sec")

        if finite_support:
            support = sphere_mask(
                shape=(
                    matrix_length,
                    matrix_length,
                    matrix_length,
                )
            )[None]
            recon = support * recon
            initial = support * initial

        # Remove the overlap that was added only to stabilize reconstruction
        # near internal temporal-segment boundaries.
        stop = n_frames - right_pad if right_pad > 0 else n_frames
        recon = recon[left_pad:stop]
        initial = initial[left_pad:stop]

        np.save(
            os.path.join(
                cache_dir,
                prefix + f"recon_sec{section}.npy",
            ),
            recon,
        )
        np.save(
            os.path.join(
                cache_dir,
                f"temp_init_sec{section}.npy",
            ),
            initial,
        )

    recon = np.concatenate(
        [
            np.load(
                os.path.join(
                    cache_dir,
                    prefix + f"recon_sec{section}.npy",
                )
            )
            for section in range(n_calc_seg)
        ],
        axis=0,
    )
    initial = np.concatenate(
        [
            np.load(
                os.path.join(
                    cache_dir,
                    f"temp_init_sec{section}.npy",
                )
            )
            for section in range(n_calc_seg)
        ],
        axis=0,
    )

    np.save(
        os.path.join(
            save_root,
            prefix + f"w{n_window}_recon.npy",
        ),
        recon,
    )
    np.save(
        os.path.join(
            save_root,
            prefix + f"w{n_window}_init.npy",
        ),
        initial,
    )


def run_recon_gridding(
    data,
    traj,
    n_calc_seg=1,
    n_window=8,
    matrix_size=(64, 64, 64),
    prefix=None,
    save_root="./",
    finite_support=False,
):
    """Run segmented density-compensated gridding and save the result.

    Args:
        data: Prepared conventional radial k-space data.
        traj: Prepared paired-spoke radial trajectory.
        n_calc_seg: Number of calculation segments.
        n_window: Window label used in output filenames.
        matrix_size: Three-dimensional reconstruction matrix size.
        prefix: Optional filename prefix for reconstruction outputs.
        save_root: Directory in which results and caches are written.
        finite_support: Whether to apply a spherical finite-support mask.
    """
    matrix_length, _, _ = matrix_size
    prefix = prefix + "_" if prefix is not None else ""

    # Preserve each radial repetition as a separate temporal frame.
    print("splitting data and trajectory into multiframe...")
    n_segment = data.shape[-1]
    data = np.stack(
        [data[..., segment] for segment in range(n_segment)],
        axis=-1,
    )
    traj = np.stack(n_segment * [traj], axis=-1)
    multiframe_data, multiframe_traj = grouping_temp(
        data,
        traj,
        n_segment=n_segment,
    )

    n_frames_total = multiframe_data.shape[0]
    frames_per_segment = n_frames_total // n_calc_seg
    pad_size = frames_per_segment // 4

    cache_dir = os.path.join(
        save_root,
        f"w{n_window}_calcseg{n_calc_seg}cache",
    )
    os.makedirs(cache_dir, exist_ok=True)

    for section in range(n_calc_seg):
        if n_calc_seg == 1:
            left_pad = 0
            right_pad = 0
            section_data = multiframe_data
            section_traj = multiframe_traj
        elif section == n_calc_seg - 1:
            left_pad = pad_size
            right_pad = 0
            section_data = multiframe_data[
                frames_per_segment * section - left_pad:
            ]
            section_traj = multiframe_traj[
                frames_per_segment * section - left_pad:
            ]
        elif section == 0:
            left_pad = 0
            right_pad = pad_size
            section_data = multiframe_data[
                :frames_per_segment * (section + 1) + right_pad
            ]
            section_traj = multiframe_traj[
                :frames_per_segment * (section + 1) + right_pad
            ]
        else:
            left_pad = pad_size
            right_pad = pad_size
            section_data = multiframe_data[
                frames_per_segment * section - left_pad:
                frames_per_segment * (section + 1) + right_pad
            ]
            section_traj = multiframe_traj[
                frames_per_segment * section - left_pad:
                frames_per_segment * (section + 1) + right_pad
            ]

        n_frames = section_data.shape[0]
        n_coils = section_data.shape[1]

        section_traj = traj_to_flat(section_traj, matrix_length / 2)
        section_data = data_to_flat(section_data)

        print(
            f"calculation segment {section + 1} / {n_calc_seg}, "
            f"n_frames = {n_frames}, left_pad = {left_pad}, "
            f"right_pad = {right_pad}"
        )

        # Use the same Pipe-Menon density-compensation calculation as the
        # ZTE-STAR reconstruction so the comparison uses matched weighting.
        print("calculating density compensation filter...")
        from sigpy.mri.dcf import pipe_menon_dcf

        report_finiteness("ksp", section_data)
        report_finiteness("traj", section_traj)

        dcf_path = os.path.join(cache_dir, f"dcf_sec{section}.npy")
        if not os.path.exists(dcf_path):
            dcf = pipe_menon_dcf(
                section_traj,
                img_shape=(
                    matrix_length,
                    matrix_length,
                    matrix_length,
                ),
                device=sp.Device(0),
            ).get()
            dcf = dcf.reshape(n_frames, 1, -1, order="C")
            dcf = dcf / np.max(abs(dcf))
            np.save(dcf_path, dcf)

        dcf = np.load(dcf_path)

        print("starting gridding reconstruction...")
        mps = np.ones(
            (
                n_coils,
                matrix_length,
                matrix_length,
                matrix_length,
            ),
            dtype=np.complex64,
        ) / n_coils

        initial, runtime = recon_gridding(
            section_data,
            mps,
            section_traj,
            dcf,
        )
        print(f"runtime: {runtime:.5f} sec")

        if finite_support:
            support = sphere_mask(
                shape=(
                    matrix_length,
                    matrix_length,
                    matrix_length,
                )
            )[None]
            initial = support * initial

        stop = n_frames - right_pad if right_pad > 0 else n_frames
        initial = initial[left_pad:stop]

        np.save(
            os.path.join(
                cache_dir,
                f"temp_init_sec{section}.npy",
            ),
            initial,
        )

    initial = np.concatenate(
        [
            np.load(
                os.path.join(
                    cache_dir,
                    f"temp_init_sec{section}.npy",
                )
            )
            for section in range(n_calc_seg)
        ],
        axis=0,
    )
    np.save(
        os.path.join(
            save_root,
            prefix + f"w{n_window}_init.npy",
        ),
        initial,
    )


def fftc(img, axis=-1):
    """Apply a centered orthonormal forward FFT.

    Args:
        img: Input array.
        axis: Axis along which to apply the transform.

    Returns:
        Complex-valued centered Fourier transform.
    """
    return np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(img),
            axis=axis,
        )
    ) / np.sqrt(img.shape[axis])


def ifftc(img, axis=-1):
    """Apply a centered orthonormal inverse FFT.

    Args:
        img: Input Fourier-domain array.
        axis: Axis along which to apply the inverse transform.

    Returns:
        Complex-valued centered inverse Fourier transform.
    """
    return np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(img),
            axis=axis,
        )
    ) * np.sqrt(img.shape[axis])


def reorder_data_to_pairspoke(data, n_adc_delay=1.5, n_seg=144):
    """Join negative and positive half spokes into full-diameter readouts.

    Args:
        data: Input array with shape `(C, Nr, Ns, T)`.
        n_adc_delay: Readout delay in sample units. Its integer part determines
            how many zero placeholders are inserted at k-space center.
        n_seg: Number of acquisition segments containing positive-spoke and
            negative-spoke blocks.

    Returns:
        Pair-spoke data with negative halves reversed, central zero
        placeholders inserted, and positive halves appended.
    """
    n_miss_point = int(n_adc_delay)
    n_coils, _, n_spokes, n_time = data.shape
    n_pro_one_frame = n_spokes // (2 * n_seg)
    stack = []

    for segment in range(n_seg):
        pos_spokes = data[
            :,
            :,
            n_pro_one_frame * (2 * segment):
            n_pro_one_frame * (2 * segment + 1),
        ]
        neg_spokes = data[
            :,
            :,
            n_pro_one_frame * (2 * segment + 1):
            n_pro_one_frame * (2 * segment + 2),
        ]
        zero_point = np.zeros(
            (
                n_coils,
                2 * n_miss_point,
                n_pro_one_frame,
                n_time,
            ),
            dtype=np.complex64,
        )

        pair_spokes = np.concatenate(
            (
                neg_spokes[:, ::-1],
                zero_point,
                pos_spokes,
            ),
            axis=1,
        )
        stack.append(pair_spokes)

    return np.concatenate(stack, axis=2)


def algeblic_rec(data, n_gap=3, over_sampling=4):
    """Fill the central dead-time gap by algebraic reconstruction.

    The operator assumes that paired spokes form one full k-space diameter.
    Missing central samples are estimated under a finite-support constraint in
    the readout domain.

    Args:
        data: Pair-spoke array with shape `(C, N, S, T)`.
        n_gap: Number of missing samples on each side of k-space center.
        over_sampling: Readout oversampling factor used to define the finite
            object-support region.

    Returns:
        Complex64 array with the same shape as `data`, with the central gap
        algebraically infilled.
    """
    n_coils, n_points, n_spokes, n_time = data.shape
    n_gap = int(n_gap)
    fov = n_points // over_sampling

    support = np.diag(
        np.array(
            [0] * ((n_points - fov) // 2)
            + [1] * fov
            + [0] * ((n_points - fov) // 2)
        )
    )
    sampling = np.diag(
        np.array(
            [1] * (n_points // 2 - n_gap)
            + [0] * (2 * n_gap)
            + [1] * (n_points // 2 - n_gap)
        )
    )

    forward_fft = lambda values: fftc(values, axis=0)
    inverse_fft = lambda values: ifftc(values, axis=0)
    pseudo_inverse = np.linalg.pinv(
        sampling @ forward_fft(support),
        1e-6,
    )
    reconstruct = lambda values: inverse_fft(
        pseudo_inverse @ values
    )

    out = np.zeros_like(data, dtype=np.complex64)
    for coil in range(n_coils):
        print(f"Processing coil {coil + 1}/{n_coils}")
        for frame in tqdm(range(n_time)):
            for spoke in range(n_spokes):
                out[coil, :, spoke, frame] = reconstruct(
                    data[coil, ::-1, spoke, frame]
                )

    return out


def prepare_pairspoke_data(
    ksp: np.ndarray,
    *,
    n_adc_delay: float = 1.5,
    n_seg: int = 144,
    n_gap: int = 5,
    over_sampling: int = 4,
):
    """Prepare ZTE-STAR k-space for temporal reconstruction.

    Args:
        ksp: Input k-space with shape `(C, Nr, Ns)` in acquisition order.
        n_adc_delay: Readout delay in sample units.
        n_seg: Number of spiral-phyllotaxis interleaves used for pairing.
        n_gap: Number of missing central samples on each side of the pair.
        over_sampling: Readout oversampling factor used by algebraic infilling.

    Returns:
        Pair-spoke k-space after reordering and algebraic dead-time infilling.
    """
    data = ksp[:, int(n_adc_delay):, :, None]
    data = reorder_data_to_pairspoke(
        data,
        n_adc_delay=n_adc_delay,
        n_seg=n_seg,
    )
    return algeblic_rec(
        data,
        n_gap=n_gap,
        over_sampling=over_sampling,
    )


def prepare_radial_repetitions(
    ksp: np.ndarray,
    *,
    n_rep: int = 16,
    n_pro: int = 2948,
    n_adc_delay: float = 1.5,
    n_gap: int = 5,
    over_sampling: int = 4,
):
    """Prepare repeated conventional radial data for gridding.

    Args:
        ksp: Continuous radial k-space with shape `(C, Nr, Ns_total)`.
        n_rep: Number of radial repetitions in the input stream.
        n_pro: Number of spokes per radial repetition.
        n_adc_delay: Readout delay in sample units.
        n_gap: Number of missing central samples on each side of the pair.
        over_sampling: Readout oversampling factor used by algebraic infilling.

    Returns:
        Radial k-space separated by repetition, reordered into paired spokes,
        and algebraically infilled.
    """
    ksp = np.stack(
        [
            ksp[..., repetition * n_pro:(repetition + 1) * n_pro]
            for repetition in range(n_rep)
        ],
        axis=-1,
    )
    data = ksp[:, int(n_adc_delay):, :]
    data = reorder_data_to_pairspoke(
        data,
        n_adc_delay=n_adc_delay,
        n_seg=1,
    )
    return algeblic_rec(
        data,
        n_gap=n_gap,
        over_sampling=over_sampling,
    )


def run_ttv_spiral(
    data,
    traj,
    *,
    M: int,
    P: int,
    n_window: int = 3,
    lamda: float = 1e-6,
    n_iter: int = 50,
    matrix_size=(64, 64, 64),
    n_calc_seg: int = 1,
    finite_support: bool = True,
    save_root="./",
    prefix: str | None = None,
):
    """Run ZTE-STAR temporal-TV reconstruction.

    Args:
        data: Prepared paired-spoke ZTE-STAR k-space.
        traj: Matching paired-spoke trajectory.
        M: Number of interleaves in one complete acquisition cycle.
        P: Number of positive spokes per interleave.
        n_window: Number of interleaves grouped into each reconstructed frame.
        lamda: Temporal total-variation regularization weight.
        n_iter: Maximum number of ADMM iterations.
        matrix_size: Three-dimensional reconstruction matrix size.
        n_calc_seg: Number of overlapping temporal calculation segments.
        finite_support: Whether to apply a spherical finite-support mask.
        save_root: Directory in which reconstruction outputs are stored.
        prefix: Optional filename prefix. If omitted, the regularization weight
            is encoded in the prefix.

    Returns:
        A tuple `(recon, initial)` containing the TTV reconstruction and the
        density-compensated adjoint initialization.
    """
    prefix = prefix if prefix is not None else f"lm{lamda:.0e}"
    save_root = str(save_root)

    run_recon(
        data,
        traj,
        n_calc_seg=n_calc_seg,
        n_window=n_window,
        M=M,
        P=P,
        lamda=lamda,
        matrix_size=list(matrix_size),
        n_iter=n_iter,
        prefix=prefix,
        save_root=save_root,
        finite_support=finite_support,
    )

    root = Path(save_root)
    recon = np.load(root / f"{prefix}_w{n_window}_recon.npy")
    initial = np.load(root / f"{prefix}_w{n_window}_init.npy")
    return recon, initial


def run_gridding(
    data,
    traj,
    *,
    n_window: int = 5,
    matrix_size=(64, 64, 64),
    n_calc_seg: int = 1,
    finite_support: bool = True,
    save_root="./",
    prefix: str = "radial",
):
    """Run conventional radial density-compensated gridding.

    Args:
        data: Prepared repeated radial k-space.
        traj: Matching paired-spoke radial trajectory.
        n_window: Window label used in the output filename.
        matrix_size: Three-dimensional reconstruction matrix size.
        n_calc_seg: Number of calculation segments.
        finite_support: Whether to apply a spherical finite-support mask.
        save_root: Directory in which reconstruction outputs are stored.
        prefix: Filename prefix for the saved gridding result.

    Returns:
        Reconstructed radial image series.
    """
    save_root = str(save_root)

    run_recon_gridding(
        data,
        traj,
        n_calc_seg=n_calc_seg,
        n_window=n_window,
        matrix_size=list(matrix_size),
        prefix=prefix,
        save_root=save_root,
        finite_support=finite_support,
    )

    return np.load(
        Path(save_root) / f"{prefix}_w{n_window}_init.npy"
    )
