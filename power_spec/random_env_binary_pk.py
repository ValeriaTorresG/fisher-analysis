import argparse, json, os, time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from pypower import CatalogFFTPower

from pk_catalogs import (DEFAULT_HOD_ROOT, apply_box_geometry_single,
                         read_hod_box_defaults, resolve_hod_paths)

plt.style.use('dark_background')
plt.rcParams.update({'grid.linewidth': 0.15,
                     'text.usetex': True})

WEBTYPES = ('void', 'sheet', 'filament', 'knot')
WEBTYPE_COLORS = {'void': 'cyan',
                  'sheet': 'orange',
                  'filament': 'limegreen',
                  'knot': 'magenta'}
NGC_RA_MIN_DEG = 90.0 #default for dr2,dr3
NGC_RA_MAX_DEG = 300.0
DEFAULT_DR2_RAW_FILE = '/pscratch/sd/v/vtorresg/cosmic-web/dr2/raw/zone_NGC_LRG.fits.gz'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-file', type=str, default=DEFAULT_DR2_RAW_FILE)
    parser.add_argument('--classification-dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dr2', choices=['dr2', 'hod'])
    parser.add_argument('--hod-root', type=str, default=DEFAULT_HOD_ROOT)
    parser.add_argument('--cosmo', type=str, default='')
    parser.add_argument('--hod-zone', type=str, default='')
    parser.add_argument('--zone', type=str, default='NGC')
    parser.add_argument('--tracer', type=str, default='LRG')
    parser.add_argument('--iterations', nargs='*', type=int, default=None)
    parser.add_argument('--iter-start', type=int, default=0)
    parser.add_argument('--n-iterations', type=int, default=100)
    parser.add_argument('--webtype', type=str, default='void', choices=WEBTYPES)
    parser.add_argument('--r-lower', type=float, default=-0.25)
    parser.add_argument('--r-med', type=float, default=0.25)
    parser.add_argument('--r-upper', type=float, default=0.65)
    parser.add_argument('--zmin', type=float, default=0.4)
    parser.add_argument('--zmax', type=float, default=1.1)
    parser.add_argument('--sky-region', type=str, default='ALL', choices=['NGC', 'SGC', 'ALL'])
    parser.add_argument('--position-source', type=str, default='cartesian', hoices=['cartesian', 'radec-z'])
    parser.add_argument('--cartesian-scale', type=float, default=0.6766)
    parser.add_argument('--h0', type=float, default=100.0)
    parser.add_argument('--om0', type=float, default=0.315)
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--mas', type=str, default='CIC', choices=['NGP', 'CIC', 'TSC', 'PCS'])
    parser.add_argument('--interlacing', type=int, default=2)
    parser.add_argument('--box-padding', type=float, default=50.0)
    parser.add_argument('--boxsize', type=float, default=0.0)
    parser.add_argument('--nthreads', type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument('--random-subsample', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--use-catalog-weights', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--subtract-shotnoise', action='store_true')
    parser.add_argument('--window-pk', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--raw-randiter-sorted', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--validate-order', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ratio-denom-min', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default='')
    return parser.parse_args()


def resolve_existing_path(path):
    path = Path(path).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        absolute = Path('/') / path
        if absolute.exists():
            return absolute.resolve()
    return path.resolve()


def resolve_outdir(user_outdir):
    if user_outdir:
        outdir = Path(user_outdir).expanduser().resolve()
    else:
        pscratch = os.environ.get('PSCRATCH')
        if pscratch:
            outdir = Path(pscratch) / 'fisher_info' / 'pypower_random_env_binary_pk'
        else:
            outdir = Path.cwd() / 'outputs' / 'pypower_random_env_binary_pk'
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def fits_colnames(path):
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
        header = hdul[1].header
        n_fields = int(header.get('TFIELDS', 0))
        return [header.get(f'TTYPE{i}') for i in range(1, n_fields + 1)]


def read_fits_columns_case_insensitive(path, columns):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = list(data.names or [])
        lookup = {name.upper(): name for name in names}
        missing = [col for col in columns if col.upper() not in lookup]
        if missing:
            raise KeyError(f'Missing columns in {path}: {missing}')

        out = {}
        for col in columns:
            out[col] = np.asarray(data[lookup[col.upper()]]).copy()
        return out


def iteration_list(args):
    if args.iterations is not None and len(args.iterations) > 0:
        return list(dict.fromkeys(int(i) for i in args.iterations))
    if args.n_iterations <= 0:
        raise ValueError()
    return list(range(args.iter_start, args.iter_start + args.n_iterations))


def classification_path(classification_dir, zone, tracer, iteration, dataset='dr2'):
    if str(dataset).lower() == 'hod':
        return Path(classification_dir) / f'zone_{zone}_iter{iteration:03d}.fits.gz'
    return Path(classification_dir) / f'zone_{zone.upper()}_{tracer.upper()}_iter{iteration:03d}.fits.gz'


def ngc_ra_mask(ra_deg):
    ra = np.asarray(ra_deg, dtype=np.float64)
    return np.isfinite(ra) & (ra > NGC_RA_MIN_DEG) & (ra < NGC_RA_MAX_DEG)


def mas_to_resampler(mas):
    mapping = {'NGP': 'ngp', 'CIC': 'cic', 'TSC': 'tsc', 'PCS': 'pcs'}
    return mapping[mas.upper()]


def radec_to_cartesian(ra_deg, dec_deg, chi):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    chi = np.asarray(chi, dtype=np.float64)
    cos_dec = np.cos(dec)
    x = chi * cos_dec * np.cos(ra)
    y = chi * cos_dec * np.sin(ra)
    z = chi * np.sin(dec)
    return np.column_stack([x, y, z]).astype(np.float32)


def compute_box_geometry(positions, boxsize_arg, box_padding):
    mins = np.min(positions, axis=0).astype(np.float64)
    maxs = np.max(positions, axis=0).astype(np.float64)
    lengths = maxs - mins

    if boxsize_arg > 0.0:
        boxsize = float(boxsize_arg)
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize
    else:
        boxsize = float(np.max(lengths) + 2.0 * box_padding)
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize

    if boxsize <= 0.0:
        raise RuntimeError('Invalid boxsize computed from input positions.')
    return boxsize, center, origin


def shift_and_clip_to_box(positions, origin, boxsize):
    shifted = positions.astype(np.float32, copy=False)
    shifted -= origin.astype(np.float32)
    upper = np.nextafter(np.float32(boxsize), np.float32(0.0))
    np.clip(shifted, np.float32(0.0), upper, out=shifted)
    return shifted


def make_k_edges(boxsize, nmesh):
    dk = 2.0 * np.pi / float(boxsize)
    k_nyquist = np.pi * float(nmesh) / float(boxsize)
    edges = np.arange(0.0, k_nyquist + dk, dk, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([0.0, max(k_nyquist, dk)], dtype=np.float64)
    return edges


def to_scalar_float(value, default=np.nan):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(np.real(arr.ravel()[0]))


def extract_monopole(poles):
    remove_shotnoise_supported = True
    try:
        pk0 = poles(ell=0, complex=False, remove_shotnoise=False)
    except TypeError:
        remove_shotnoise_supported = False
        pk0 = poles(ell=0, complex=False)

    k = np.asarray(poles.k, dtype=np.float64)
    pk0 = np.asarray(pk0, dtype=np.float64)
    nmodes = getattr(poles, 'nmodes', None)
    if nmodes is None:
        nmodes = np.full_like(k, np.nan, dtype=np.float64)
    else:
        nmodes = np.asarray(nmodes, dtype=np.float64)
    shotnoise = to_scalar_float(getattr(poles, 'shotnoise', np.nan), default=np.nan)
    return {'k': k, 'pk0': pk0, 'nmodes': nmodes, 'shotnoise': shotnoise,
            'remove_shotnoise_supported': remove_shotnoise_supported}


def compute_pk_pypower(data_positions, edges, boxsize, boxcenter, nmesh, resampler, interlacing,
                       data_weights=None, random_positions=None, random_weights=None):
    kwargs = {'data_positions1': data_positions,
              'edges': edges,
              'ells': (0,),
              'position_type': 'pos',
              'boxsize': boxsize,
              'boxcenter': boxcenter,
              'nmesh': nmesh,
              'resampler': resampler,
              'interlacing': interlacing}
    if data_weights is not None:
        kwargs['data_weights1'] = data_weights
    if random_positions is not None:
        kwargs['randoms_positions1'] = random_positions
        if random_weights is not None:
            kwargs['randoms_weights1'] = random_weights
    result = CatalogFFTPower(**kwargs)
    return extract_monopole(result.poles)


def align_to_k(k_target, k_src, values_src):
    k_target = np.asarray(k_target, dtype=np.float64)
    k_src = np.asarray(k_src, dtype=np.float64)
    values_src = np.asarray(values_src, dtype=np.float64)
    if k_target.shape == k_src.shape and np.allclose(k_target, k_src, rtol=1e-8, atol=1e-12):
        return values_src.copy()
    order = np.argsort(k_src)
    return np.interp(k_target, k_src[order], values_src[order], left=np.nan, right=np.nan)


def classify_webtype_from_counts(ndata, nrand, r_lower, r_med, r_upper):
    ndata = np.asarray(ndata, dtype=np.float64)
    nrand = np.asarray(nrand, dtype=np.float64)
    denom = ndata + nrand
    ratio = np.full(denom.shape, np.nan, dtype=np.float64)
    np.divide(ndata - nrand, denom, out=ratio, where=denom > 0.0)

    class_index = np.full(denom.shape, -1, dtype=np.int16)
    valid = np.isfinite(ratio)
    if np.any(valid):
        bins = np.array([r_lower, r_med, r_upper], dtype=np.float64)
        class_index[valid] = np.clip(np.digitize(ratio[valid], bins, right=False), 0, 3).astype(np.int16)
    return class_index, ratio


def read_raw_randoms(raw_file, iterations, args):
    available = set(name.upper() for name in fits_colnames(raw_file))
    required = ['TARGETID', 'RANDITER', 'TRACER_ID']
    if args.position_source == 'cartesian':
        required.extend(['XCART', 'YCART', 'ZCART'])
    else:
        required.extend(['RA', 'DEC', 'Z'])

    need_z = args.zmin > 0.0 or args.zmax > 0.0
    if need_z and 'Z' not in required:
        required.append('Z')
    if args.sky_region in ('NGC', 'SGC') and 'RA' not in required:
        required.append('RA')

    use_catalog_weights = bool(args.use_catalog_weights and 'WEIGHT' in available)
    if use_catalog_weights:
        required.append('WEIGHT')
    elif args.use_catalog_weights:
        print('---> raw WEIGHT column not found; using unit weights.')

    raw = read_fits_columns_case_insensitive(str(raw_file), required)
    randiter = np.asarray(raw['RANDITER'], dtype=np.int32)

    if len(iterations) == 0:
        raise ValueError('No iterations requested.')
    if iterations == list(range(min(iterations), max(iterations) + 1)):
        m_iter = (randiter >= min(iterations)) & (randiter <= max(iterations))
    else:
        m_iter = np.isin(randiter, np.asarray(iterations, dtype=np.int32))
    mask = (randiter >= 0) & m_iter

    if need_z:
        z = np.asarray(raw['Z'], dtype=np.float64)
        if args.zmin > 0.0:
            mask &= z >= args.zmin
        if args.zmax > 0.0:
            mask &= z < args.zmax
    if args.sky_region == 'NGC':
        mask &= ngc_ra_mask(raw['RA'])
    elif args.sky_region == 'SGC':
        mask &= np.isfinite(np.asarray(raw['RA'], dtype=np.float64)) & ~ngc_ra_mask(raw['RA'])

    out = {}
    for col, values in raw.items():
        out[col] = np.asarray(values[mask])
    del raw

    if len(out['TARGETID']) == 0:
        raise RuntimeError('No raw random rows selected after cuts---')

    weights = None
    if use_catalog_weights:
        weights = np.asarray(out['WEIGHT'], dtype=np.float64)
        if np.any(~np.isfinite(weights)):
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)

    return {'targetid': np.asarray(out['TARGETID'], dtype=np.int64),
            'randiter': np.asarray(out['RANDITER'], dtype=np.int32),
            'tracer_id': np.asarray(out['TRACER_ID'], dtype=np.int16),
            'columns': out,
            'weights': weights,
            'use_catalog_weights': use_catalog_weights}


def raw_iteration_selector(raw_randiter, iteration, assume_sorted):
    if assume_sorted:
        start = int(np.searchsorted(raw_randiter, iteration, side='left'))
        end = int(np.searchsorted(raw_randiter, iteration, side='right'))
        return slice(start, end), end - start
    idx = np.nonzero(raw_randiter == iteration)[0]
    return idx, idx.size


def selector_to_indices(selector, size):
    if isinstance(selector, slice):
        return np.arange(selector.start, selector.stop, dtype=np.int64)
    return np.asarray(selector, dtype=np.int64)


def key_align_iteration(raw_targetid, raw_randiter, raw_tracer_id,
                        class_targetid, class_randiter, class_tracer_id,
                        raw_selector, class_selected):
    raw_indices = selector_to_indices(raw_selector, raw_targetid.size)
    key_dtype = np.dtype([('TARGETID', np.int64),
                          ('RANDITER', np.int32),
                          ('TRACER_ID', np.int16)])
    raw_keys = np.empty(raw_indices.size, dtype=key_dtype)
    raw_keys['TARGETID'] = raw_targetid[raw_indices]
    raw_keys['RANDITER'] = raw_randiter[raw_indices]
    raw_keys['TRACER_ID'] = raw_tracer_id[raw_indices]

    cls_keys = np.empty(int(np.sum(class_selected)), dtype=key_dtype)
    cls_keys['TARGETID'] = class_targetid[class_selected]
    cls_keys['RANDITER'] = class_randiter[class_selected]
    cls_keys['TRACER_ID'] = class_tracer_id[class_selected]

    sorter = np.argsort(raw_keys, order=('TARGETID', 'RANDITER', 'TRACER_ID'))
    raw_sorted = raw_keys[sorter]
    pos = np.searchsorted(raw_sorted, cls_keys)
    within = pos < raw_sorted.size
    matched = np.zeros(cls_keys.size, dtype=bool)
    if np.any(within):
        matched[within] = raw_sorted[pos[within]] == cls_keys[within]
    return raw_indices[sorter[pos[matched]]], int(np.sum(matched)), int(cls_keys.size)


def build_binary_environment_mask(raw_targetid, raw_randiter, raw_tracer_id,
                                  class_paths, iterations, args):
    target_class = WEBTYPES.index(args.webtype)
    env_mask = np.zeros(raw_targetid.size, dtype=bool)
    iteration_stats = []
    total_class_random = 0
    total_class_selected = 0
    total_matched_selected = 0
    n_order_matches = 0
    n_key_fallbacks = 0

    for iteration, path in zip(iterations, class_paths):
        print(f'---> reading classification iter={iteration:03d}: {path}')
        cls = read_fits_columns_case_insensitive(str(path),
                                                 ['TARGETID', 'RANDITER', 'ISDATA',
                                                  'NDATA', 'NRAND', 'TRACER_ID'])
        class_randiter = np.asarray(cls['RANDITER'], dtype=np.int32)
        class_is_random = ~np.asarray(cls['ISDATA'], dtype=bool)
        class_is_random &= class_randiter == int(iteration)

        class_index, ratio = classify_webtype_from_counts(cls['NDATA'], cls['NRAND'],
                                                          args.r_lower, args.r_med, args.r_upper)
        class_random_index = class_index[class_is_random]
        class_web_random = class_random_index == target_class
        n_class_random = int(np.sum(class_is_random))
        n_class_web = int(np.sum(class_web_random))

        raw_selector, n_raw_iter = raw_iteration_selector(raw_randiter, int(iteration),
                                                          args.raw_randiter_sorted)

        used_order = False
        matched_selected = 0
        if args.validate_order and n_raw_iter == n_class_random and n_raw_iter > 0:
            class_target_random = np.asarray(cls['TARGETID'], dtype=np.int64)[class_is_random]
            class_tracer_random = np.asarray(cls['TRACER_ID'], dtype=np.int16)[class_is_random]
            raw_target_view = raw_targetid[raw_selector]
            raw_tracer_view = raw_tracer_id[raw_selector]
            same_target = np.array_equal(raw_target_view, class_target_random)
            same_tracer = np.array_equal(raw_tracer_view, class_tracer_random)
            if same_target and same_tracer:
                env_mask[raw_selector] = class_web_random
                matched_selected = n_class_web
                used_order = True
                n_order_matches += 1

        if not used_order:
            n_key_fallbacks += 1
            class_selected = class_is_random & (class_index == target_class)
            matched_indices, matched_selected, selected_size = key_align_iteration(
                raw_targetid=raw_targetid,
                raw_randiter=raw_randiter,
                raw_tracer_id=raw_tracer_id,
                class_targetid=np.asarray(cls['TARGETID'], dtype=np.int64),
                class_randiter=class_randiter,
                class_tracer_id=np.asarray(cls['TRACER_ID'], dtype=np.int16),
                raw_selector=raw_selector,
                class_selected=class_selected)
            env_mask[matched_indices] = True
            if selected_size != n_class_web:
                raise RuntimeError('selection mismatch in classification rows')

        total_class_random += n_class_random
        total_class_selected += n_class_web
        total_matched_selected += matched_selected
        valid_ratio_random = ratio[class_is_random]
        iteration_stats.append({'iteration': int(iteration),
                                'path': str(path),
                                'n_raw_random': int(n_raw_iter),
                                'n_class_random': n_class_random,
                                f'n_{args.webtype}_class_random': n_class_web,
                                f'n_{args.webtype}_matched_raw': int(matched_selected),
                                f'fraction_{args.webtype}_class_random': (
                                    float(n_class_web / n_class_random) if n_class_random > 0 else np.nan),
                                'ratio_min_random': float(np.nanmin(valid_ratio_random)) if valid_ratio_random.size else np.nan,
                                'ratio_max_random': float(np.nanmax(valid_ratio_random)) if valid_ratio_random.size else np.nan,
                                'alignment': 'order' if used_order else 'key'})
        del cls #avoid memory issues? might be better

    info = {'n_raw_random': int(raw_targetid.size),
            'n_class_random_total': int(total_class_random),
            f'n_{args.webtype}_class_random_total': int(total_class_selected),
            f'n_{args.webtype}_matched_raw_total': int(total_matched_selected),
            f'fraction_{args.webtype}_of_class_random': (
                float(total_class_selected / total_class_random) if total_class_random > 0 else np.nan),
            'n_order_aligned_iterations': int(n_order_matches),
            'n_key_fallback_iterations': int(n_key_fallbacks),
            'iteration_stats': iteration_stats}
    return env_mask, info


def build_positions(raw_columns, args):
    if args.position_source == 'cartesian':
        pos = np.column_stack([np.asarray(raw_columns['XCART'], dtype=np.float32),
                               np.asarray(raw_columns['YCART'], dtype=np.float32),
                               np.asarray(raw_columns['ZCART'], dtype=np.float32)])
        if args.cartesian_scale != 1.0:
            pos *= np.float32(args.cartesian_scale)
        return pos.astype(np.float32, copy=False)

    cosmo = FlatLambdaCDM(H0=args.h0, Om0=args.om0)
    chi = np.asarray(cosmo.comoving_distance(raw_columns['Z']).value, dtype=np.float64)
    return radec_to_cartesian(raw_columns['RA'], raw_columns['DEC'], chi)


def finite_positive_sum(weights, n_rows, label):
    if weights is None:
        return float(n_rows), float(n_rows)
    sw = float(np.sum(weights, dtype=np.float64))
    sw2 = float(np.sum(weights.astype(np.float64) ** 2, dtype=np.float64))
    if sw <= 0.0:
        raise RuntimeError(f'Sum of {label} weights is <= 0.')
    return sw, sw2


def main():
    args = parse_args()
    if not (args.r_lower < args.r_med < args.r_upper):
        raise ValueError()
    iterations = iteration_list(args)
    dataset = str(args.dataset or 'dr2').lower()
    box_defaults = None
    if dataset == 'hod':
        if str(args.raw_file or '').strip() == DEFAULT_DR2_RAW_FILE:
            args.raw_file = ''
        if args.zmin != 0.0 or args.zmax != 0.0:
            print('---> HOD dataset: ignoring z cuts because HOD raw catalogs have no redshift column.')
            args.zmin = 0.0
            args.zmax = 0.0
        raw_file_resolved, _, hod_zone, _ = resolve_hod_paths(args)
        args.raw_file = raw_file_resolved
        args.zone = hod_zone
        args.tracer = 'HOD'
        if args.cartesian_scale == 0.6766:
            args.cartesian_scale = 1.0
        raw_file = resolve_existing_path(args.raw_file)
        if args.classification_dir:
            class_dir = resolve_existing_path(args.classification_dir)
        else:
            class_dir = resolve_existing_path(Path(raw_file_resolved).parent.parent
                                              / 'release' / 'classification')
        box_defaults = read_hod_box_defaults(raw_file, scale=args.cartesian_scale)
    else:
        raw_file = resolve_existing_path(args.raw_file)
        if args.classification_dir:
            class_dir = resolve_existing_path(args.classification_dir)
        else:
            class_dir = resolve_existing_path(
                Path('/pscratch/sd/v/vtorresg/cosmic-web/dr2/classification')
                / args.tracer.lower() / args.zone.lower())

    class_paths = [classification_path(class_dir, args.zone, args.tracer, iteration, dataset=dataset)
                   for iteration in iterations]
    missing = [str(path) for path in class_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing classification files: ' + ', '.join(missing[:5])
                                + (' ' if len(missing) > 5 else ''))

    outdir = resolve_outdir(args.outdir)
    iter_tag = f'i{min(iterations):03d}-{max(iterations):03d}_n{len(iterations):03d}'
    tag = (f'{args.tracer}_{args.zone}_{args.webtype}_random_'
           f'z{args.zmin:.3f}_{args.zmax:.3f}_{iter_tag}_N{args.grid}')
    t0 = time.time()

    print(f'---> raw ASTRA catalog: {raw_file}')
    print(f'---> classification dir: {class_dir}')
    print(f'---> dataset: {dataset}')
    print(f'---> iterations: {iterations[:8]}'
          + (' ' if len(iterations) > 8 else ''))
    print(f'---> webtype: {args.webtype}')
    print(f'---> r thresholds: {args.r_lower}, {args.r_med}, {args.r_upper}')
    print(f'---> position source: {args.position_source}')
    if args.position_source == 'cartesian':
        print(f'---> cartesian scale:  {args.cartesian_scale}')
    print(f'---> outdir: {outdir}')
    print(f'---> nmesh: {args.grid}')
    print(f'---> mas/resampler: {args.mas}/{mas_to_resampler(args.mas)}')
    print(f'---> interlacing: {args.interlacing}')
    print(f'---> nthreads hint: {args.nthreads}')

    # print('---> loading raw random rows--------')
    raw = read_raw_randoms(raw_file, iterations, args)
    n_random_before_subsample = len(raw['targetid'])
    print(f'---> selected raw random rows: {n_random_before_subsample}')

    print(f'---> building binary {args.webtype} mask from classification files ')
    env_mask, class_info = build_binary_environment_mask(raw_targetid=raw['targetid'],
                                                         raw_randiter=raw['randiter'],
                                                         raw_tracer_id=raw['tracer_id'],
                                                         class_paths=class_paths,
                                                         iterations=iterations,
                                                         args=args)

    if args.random_subsample < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.random(len(env_mask)) < args.random_subsample
        for key in ('targetid', 'randiter', 'tracer_id'):
            raw[key] = raw[key][keep]
        for col in list(raw['columns']):
            raw['columns'][col] = raw['columns'][col][keep]
        env_mask = env_mask[keep]
        if raw['weights'] is not None:
            raw['weights'] = raw['weights'][keep]
        print(f'---> random subsample fraction={args.random_subsample:.3f}, kept={len(env_mask)}')

    n_random = int(len(env_mask))
    n_env = int(np.sum(env_mask))
    if n_random == 0:
        raise RuntimeError('No random rows remain after subsampling')
    if n_env == 0:
        raise RuntimeError(f'No random rows classified as {args.webtype}')
    print(f'---> {args.webtype} random rows: {n_env} / {n_random} ({n_env / n_random:.6f})')

    print('---> building positions ')
    pos_r = build_positions(raw['columns'], args)
    weights_r = raw['weights']
    weights_env = weights_r[env_mask] if weights_r is not None else None
    pos_env = pos_r[env_mask]

    pos_r, boxsize, center, origin, boxcenter = apply_box_geometry_single(
        pos_r, args=args, catalog=box_defaults)
    pos_env = shift_and_clip_to_box(pos_env, origin=origin, boxsize=boxsize)
    volume = boxsize ** 3
    print(f'---> boxsize [Mpc/h]: {boxsize:.3f}')

    sw_r, sw2_r = finite_positive_sum(weights_r, n_random, 'random')
    sw_env, sw2_env = finite_positive_sum(weights_env, n_env, args.webtype)
    alpha_env = sw_env / sw_r

    resampler = mas_to_resampler(args.mas)
    edges = make_k_edges(boxsize=boxsize, nmesh=args.grid)

    print(f'---> computing P_{args.webtype}_random(k) with PyPower ')
    pk_env_result = compute_pk_pypower(data_positions=pos_env,
                                       data_weights=weights_env,
                                       random_positions=pos_r,
                                       random_weights=weights_r,
                                       edges=edges,
                                       boxsize=boxsize,
                                       boxcenter=boxcenter,
                                       nmesh=args.grid,
                                       resampler=resampler,
                                       interlacing=args.interlacing)
    k = pk_env_result['k']
    pk_env_raw = pk_env_result['pk0']
    nmodes = pk_env_result['nmodes']
    shotnoise_env_analytic = volume * sw2_env / (sw_env * sw_env)
    shotnoise_env = (pk_env_result['shotnoise']
                     if np.isfinite(pk_env_result['shotnoise'])
                     else shotnoise_env_analytic)
    pk_env_used = pk_env_raw - shotnoise_env if args.subtract_shotnoise else pk_env_raw.copy()

    pk_window = np.full_like(pk_env_used, np.nan)
    shotnoise_window = np.nan
    shotnoise_window_pypower = np.nan
    if args.window_pk:
        print('---> computing P_window(k) from all selected raw randoms')
        pk_window_result = compute_pk_pypower(data_positions=pos_r,
                                              data_weights=weights_r,
                                              edges=edges,
                                              boxsize=boxsize,
                                              boxcenter=boxcenter,
                                              nmesh=args.grid,
                                              resampler=resampler,
                                              interlacing=args.interlacing)
        pk_window = align_to_k(k, pk_window_result['k'], pk_window_result['pk0'])
        shotnoise_window_analytic = volume * sw2_r / (sw_r * sw_r)
        shotnoise_window_pypower = pk_window_result['shotnoise']
        shotnoise_window = (shotnoise_window_pypower
                            if np.isfinite(shotnoise_window_pypower)
                            else shotnoise_window_analytic)
        if args.subtract_shotnoise:
            pk_window = pk_window - shotnoise_window

    ratio_env_window = np.full_like(pk_env_used, np.nan)
    good_ratio = (np.isfinite(pk_env_used)
                  & np.isfinite(pk_window)
                  & (np.abs(pk_window) > args.ratio_denom_min))
    ratio_env_window[good_ratio] = pk_env_used[good_ratio] / pk_window[good_ratio]

    csv_path = outdir / f'pk_random_env_binary_{tag}.csv'
    fig_path = outdir / f'pk_random_env_binary_{tag}.png'
    fig_ratio_path = outdir / f'pk_random_env_over_window_{tag}.png'
    meta_path = outdir / f'run_metadata_pk_random_env_binary_{tag}.json'

    np.savetxt(csv_path,
               np.column_stack([k, pk_env_raw, pk_env_used, nmodes, pk_window, ratio_env_window]),
               delimiter=',',
               header=(f'k_h_mpc,pk_{args.webtype}_random_raw,pk_{args.webtype}_random_used,'
                       'nmodes,pk_random_window,pk_random_env_over_window'),
               comments='')

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m_env = (k > 0.0) & np.isfinite(pk_env_used) & (pk_env_used > 0.0)
    ax.loglog(k[m_env], pk_env_used[m_env], lw=1.5,
              color=WEBTYPE_COLORS.get(args.webtype, 'cyan'),
              label=fr'{args.webtype} randoms')
    if args.window_pk:
        m_win = (k > 0.0) & np.isfinite(pk_window) & (pk_window > 0.0)
        ax.loglog(k[m_win], pk_window[m_win], lw=1.2, ls='--',
                  color='white', label='all randoms window')
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    title = f'{args.tracer} random {args.webtype} field ({len(iterations)} ASTRA iterations)'
    if args.subtract_shotnoise:
        title += '  [shot-noise subtracted]'
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    if args.window_pk:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        m_ratio = (k > 0.0) & np.isfinite(ratio_env_window)
        ax.semilogx(k[m_ratio], ratio_env_window[m_ratio], lw=1.5,
                    color=WEBTYPE_COLORS.get(args.webtype, 'cyan'))
        ax.axhline(1.0, color='white', ls='--', lw=1.0, alpha=0.7)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(fr'$P_{{\rm {args.webtype},rand}}(k) / P_{{\rm window}}(k)$')
        ax.set_title(f'{args.tracer} random {args.webtype} / random window')
        ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(fig_ratio_path, dpi=300)
        plt.close(fig)

    elapsed = time.time() - t0
    metadata = {'tracer': args.tracer,
                'dataset': dataset,
                'zone': args.zone,
                'webtype': args.webtype,
                'raw_file': str(raw_file),
                'classification_dir': str(class_dir),
                'classification_paths': [str(path) for path in class_paths],
                'iterations': iterations,
                'n_iterations': len(iterations),
                'r_thresholds': {'r_lower': args.r_lower,
                                 'r_med': args.r_med,
                                 'r_upper': args.r_upper},
                'zmin': args.zmin,
                'zmax': args.zmax,
                'sky_region': args.sky_region,
                'position_source': args.position_source,
                'cartesian_scale': args.cartesian_scale,
                'h0': args.h0,
                'om0': args.om0,
                'n_random_before_subsample': n_random_before_subsample,
                'n_random': n_random,
                f'n_random_{args.webtype}': n_env,
                f'fraction_random_{args.webtype}': float(n_env / n_random),
                'classification_info': class_info,
                'use_catalog_weights': raw['use_catalog_weights'],
                'sum_w_random': sw_r,
                f'sum_w_{args.webtype}_random': sw_env,
                f'alpha_{args.webtype}_random': alpha_env,
                'grid': args.grid,
                'mas': args.mas,
                'resampler': resampler,
                'interlacing': args.interlacing,
                'boxsize_mpc_h': boxsize,
                'box_center_xyz_mpc_h': center.tolist(),
                'box_padding_mpc_h': args.box_padding,
                'volume_mpc3_h3': volume,
                'subtract_shotnoise': args.subtract_shotnoise,
                f'shotnoise_{args.webtype}_random': shotnoise_env,
                f'shotnoise_{args.webtype}_random_analytic': shotnoise_env_analytic,
                f'shotnoise_{args.webtype}_random_pypower': pk_env_result['shotnoise'],
                'window_pk': args.window_pk,
                'shotnoise_window': shotnoise_window,
                'shotnoise_window_pypower': shotnoise_window_pypower,
                'ratio_denom_min': args.ratio_denom_min,
                'raw_randiter_sorted': args.raw_randiter_sorted,
                'validate_order': args.validate_order,
                'random_subsample': args.random_subsample,
                'nthreads_hint': args.nthreads,
                'elapsed_sec': elapsed,
                'outputs': {'pk_csv': str(csv_path),
                            'pk_plot': str(fig_path),
                            'pk_random_env_over_window_plot': str(fig_ratio_path) if args.window_pk else None},
                'engine': 'pypower.CatalogFFTPower',
                'remove_shotnoise_flag_supported': bool(pk_env_result['remove_shotnoise_supported']),
                'notes': ['Uses only ASTRA random rows (ISDATA=False) from split classification files.',
                          'Binary environment weights are 1 for the requested webtype and 0 otherwise.',
                          'The data catalog for PyPower is the selected webtype-random subset; '
                          'the random catalog is all selected ASTRA random rows.']}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'---> wrote: {csv_path}')
    print(f'---> wrote: {fig_path}')
    if args.window_pk:
        print(f'---> wrote: {fig_ratio_path}')
    print(f'---> wrote: {meta_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()