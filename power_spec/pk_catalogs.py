import glob, re
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack


DEFAULT_DR2_LSS_BASE = '/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/'
DEFAULT_DR2_ASTRA_ROOT = '/pscratch/sd/v/vtorresg/cosmic-web/dr2'
DEFAULT_HOD_ROOT = '/pscratch/sd/v/vtorresg/hod-astra-fullbox'
DEFAULT_DR2_PROB_FILE = ('/pscratch/sd/v/vtorresg/cosmic-web/dr2/probabilities/lrg/ngc/'
                         'zone_NGC_LRG_probability_iterdata.fits.gz')
NGC_RA_MIN_DEG = 90.0 #default for dr2,dr3
NGC_RA_MAX_DEG = 300.0


class PreparedCatalog:
    def __init__(self, dataset, tracer, zone, data_path, random_paths,
                 random_indices_used, astra_prob_file, pos_d, pos_r, w_d, w_r,
                 targetid_d, n_data, n_random, n_data_before_selection,
                 n_random_before_selection, sky_region_desc, tag,
                 boxsize_default=None, box_origin_default=None,
                 box_center_default=None, metadata=None):
        self.dataset = dataset
        self.tracer = tracer
        self.zone = zone
        self.data_path = data_path
        self.random_paths = random_paths
        self.random_indices_used = random_indices_used
        self.astra_prob_file = astra_prob_file
        self.pos_d = pos_d
        self.pos_r = pos_r
        self.w_d = w_d
        self.w_r = w_r
        self.targetid_d = targetid_d
        self.n_data = n_data
        self.n_random = n_random
        self.n_data_before_selection = n_data_before_selection
        self.n_random_before_selection = n_random_before_selection
        self.sky_region_desc = sky_region_desc
        self.tag = tag
        self.boxsize_default = boxsize_default
        self.box_origin_default = box_origin_default
        self.box_center_default = box_center_default
        self.metadata = metadata or {}


class BoxGeometryDefaults:
    def __init__(self, boxsize_default=None, box_origin_default=None, box_center_default=None):
        self.boxsize_default = boxsize_default
        self.box_origin_default = box_origin_default
        self.box_center_default = box_center_default


def read_hod_box_defaults(raw_file, scale=1.0):
    with fits.open(raw_file, memmap=False, lazy_load_hdus=True) as hdul:
        header = hdul[1].header
        boxsize = header.get('HODBOX', header.get('LBOX', header.get('BOXSZX', None)))
        box_min = header.get('BOXMIN', None)
        box_max = header.get('BOXMAX', None)

    if boxsize is None or box_min is None or box_max is None:
        return BoxGeometryDefaults()
    boxsize = float(boxsize) * float(scale)
    box_min = float(box_min) * float(scale)
    box_max = float(box_max) * float(scale)
    return BoxGeometryDefaults(boxsize_default=boxsize,
                               box_origin_default=np.array([box_min, box_min, box_min], dtype=np.float64),
                               box_center_default=np.array([0.5 * (box_min + box_max)] * 3, dtype=np.float64))


def add_catalog_arguments(parser):
    parser.add_argument('--dataset', type=str, default='dr2', choices=['dr2', 'hod'])
    parser.add_argument('--dr2-astra-root', type=str, default=DEFAULT_DR2_ASTRA_ROOT)
    parser.add_argument('--hod-root', type=str, default=DEFAULT_HOD_ROOT)
    parser.add_argument('--cosmo', type=str, default='')
    parser.add_argument('--hod', type=str, default='')
    parser.add_argument('--phase', type=str, default='ph000')
    parser.add_argument('--seed-token', type=str, default='seed0')
    parser.add_argument('--hod-zone', type=str, default='')
    parser.add_argument('--raw-file', type=str, default='')
    parser.add_argument('--position-source', type=str, default='auto', choices=['auto', 'radec-z', 'cartesian'])
    parser.add_argument('--cartesian-cols', nargs=3, default=['XCART', 'YCART', 'ZCART'], metavar=('XCOL', 'YCOL', 'ZCOL'))
    parser.add_argument('--cartesian-scale', type=float, default=1.0)


def resolve_existing_path(path):
    path = Path(path).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        absolute = Path('/') / path
        if absolute.exists():
            return absolute.resolve()
    return path.resolve()


def read_fits_columns_case_insensitive(path, columns, optional_columns=()):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = list(data.names or [])
        lookup = {name.upper(): name for name in names}
        missing = [col for col in columns if col.upper() not in lookup]
        if missing:
            raise KeyError()

        out = {}
        for col in list(columns) + list(optional_columns):
            key = lookup.get(col.upper())
            if key is not None:
                out[col] = np.asarray(data[key]).copy()
        return Table(out)


def get_weight_column(table):
    if 'WEIGHT' in table.colnames:
        weights = np.asarray(table['WEIGHT'], dtype=np.float64)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(weights, 0.0, None)
    return np.ones(len(table), dtype=np.float64)


def radec_to_cartesian(ra_deg, dec_deg, chi):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    chi = np.asarray(chi, dtype=np.float64)
    cos_dec = np.cos(dec)
    x = chi * cos_dec * np.cos(ra)
    y = chi * cos_dec * np.sin(ra)
    z = chi * np.sin(dec)
    return np.column_stack([x, y, z]).astype(np.float64)


def sky_region_mask(ra_deg, region):
    region = str(region or 'ALL').upper()
    ra = np.asarray(ra_deg, dtype=np.float64)
    finite = np.isfinite(ra)
    ngc = finite & (ra >= NGC_RA_MIN_DEG) & (ra <= NGC_RA_MAX_DEG)
    if region == 'NGC':
        return ngc
    if region == 'SGC':
        return finite & ~ngc
    if region == 'ALL':
        return finite
    raise ValueError()


def sky_region_description(region):
    region = str(region or 'ALL').upper()
    if region == 'NGC':
        return f'NGC sky ({NGC_RA_MIN_DEG:.0f} <= RA <= {NGC_RA_MAX_DEG:.0f} deg)'
    if region == 'SGC':
        return f'SGC sky (RA < {NGC_RA_MIN_DEG:.0f} or RA > {NGC_RA_MAX_DEG:.0f} deg)'
    return 'ALL sky'


def load_and_stack_randoms(base_dir, tracer, start_index, n_random_files, columns, zmin, zmax,
                           optional_columns=()):
    indices = list(range(start_index, start_index + n_random_files))
    paths = [Path(base_dir) / f'{tracer}_{idx}_clustering.ran.fits' for idx in indices]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError()

    random_tables = []
    for path in paths:
        rand_i = read_fits_columns_case_insensitive(str(path), columns, optional_columns=optional_columns)
        m_i = (rand_i['Z'] >= zmin) & (rand_i['Z'] < zmax)
        random_tables.append(rand_i[m_i])

    rand = random_tables[0] if len(random_tables) == 1 else vstack(random_tables)
    return rand, indices, [str(p) for p in paths]


def tracer_probability_token(tracer):
    return str(tracer).upper().split('_', 1)[0]


def tracer_probability_subdir(tracer):
    return tracer_probability_token(tracer).lower()


def _is_default_dr2_probability(path):
    return str(path or '').strip() in ('', DEFAULT_DR2_PROB_FILE)


def resolve_dr2_probability_file(args):
    user_path = str(getattr(args, 'astra_prob_file', '') or '').strip()
    if user_path and not _is_default_dr2_probability(user_path):
        return str(resolve_existing_path(user_path))

    zone = str(getattr(args, 'sky_region', 'NGC') or 'NGC').upper()
    if zone == 'ALL':
        raise ValueError()
    tracer_token = tracer_probability_token(args.tracer)
    prob_path = (Path(args.dr2_astra_root) / 'probabilities' / tracer_probability_subdir(args.tracer)
                 / zone.lower() / f'zone_{zone}_{tracer_token}_probability_iterdata.fits.gz')
    return str(resolve_existing_path(prob_path))


def _zone_from_raw_path(path):
    name = Path(path).name
    match = re.match(r'^zone_(?P<zone>.+?)\.fits(?:\.gz)?$', name)
    return match.group('zone') if match else ''


def _cosmo_from_zone(zone):
    text = str(zone or '')
    return text.split('_', 1)[0] if text else ''


def _token_from_zone(zone, prefix):
    prefix = str(prefix)
    for token in str(zone or '').split('_'):
        if token.startswith(prefix):
            return token
    return ''


def _hod_from_zone(zone):
    return _token_from_zone(zone, 'hod')


def _phase_from_zone(zone):
    return _token_from_zone(zone, 'ph')


def _seed_token_from_zone(zone):
    return _token_from_zone(zone, 'seed')


def normalize_cosmo_token(token):
    text = str(token or '').strip()
    if not text:
        return ''
    lowered = text.lower()
    if lowered.startswith('c') and lowered[1:].isdigit():
        return f'c{int(lowered[1:]):03d}'
    if text.isdigit():
        return f'c{int(text):03d}'
    return text


def normalize_hod_token(token):
    text = str(token or '').strip()
    if not text:
        return ''
    lowered = text.lower()
    if lowered.startswith('hod') and lowered[3:].isdigit():
        return f'hod{int(lowered[3:]):03d}'
    if text.isdigit():
        return f'hod{int(text):03d}'
    raise ValueError(f'Invalid HOD token: {token}')


def normalize_phase_token(token):
    text = str(token or '').strip()
    if not text:
        return 'ph000'
    lowered = text.lower()
    if lowered.startswith('ph') and lowered[2:].isdigit():
        return f'ph{int(lowered[2:]):03d}'
    if text.isdigit():
        return f'ph{int(text):03d}'
    return text


def normalize_seed_token(token):
    text = str(token or '').strip()
    if not text:
        return 'seed0'
    lowered = text.lower()
    if lowered.startswith('seed') and lowered[4:].isdigit():
        return f'seed{int(lowered[4:])}'
    if text.isdigit():
        return f'seed{int(text)}'
    return text


def build_hod_zone_label(cosmo, hod, phase='ph000', seed_token='seed0'):
    cosmo = normalize_cosmo_token(cosmo)
    hod = normalize_hod_token(hod)
    if not cosmo:
        raise ValueError()
    return f'{cosmo}_{normalize_phase_token(phase)}_{normalize_seed_token(seed_token)}_{hod}'


def _run_dir_from_raw_path(raw_file):
    raw_path = Path(raw_file).expanduser().resolve()
    if raw_path.parent.name == 'raw':
        return raw_path.parent.parent
    return raw_path.parent


def _raw_files_in_run_dir(run_dir, hod_zone=''):
    raw_dir = Path(run_dir) / 'raw'
    if hod_zone:
        candidates = [raw_dir / f'zone_{hod_zone}.fits',
                      raw_dir / f'zone_{hod_zone}.fits.gz']
        return [path for path in candidates if path.is_file()]
    existing = [Path(path) for path in sorted(glob.glob(str(raw_dir / 'zone_*.fits')))]
    existing += [Path(path) for path in sorted(glob.glob(str(raw_dir / 'zone_*.fits.gz')))]
    return existing


def _candidate_hod_run_dirs(hod_root, cosmo='', hod_zone=''):
    hod_root = Path(hod_root).expanduser()
    candidates = []
    seen = set()

    def add(path):
        path = Path(path)
        key = str(path)
        if key not in seen:
            candidates.append(path)
            seen.add(key)

    if cosmo:
        add(hod_root / cosmo)
    if hod_zone:
        add(hod_root / hod_zone)
    if cosmo:
        for path in sorted(hod_root.glob(f'{cosmo}_*')):
            if path.is_dir():
                add(path)
    return candidates


def resolve_hod_paths(args):
    raw_file_arg = str(getattr(args, 'raw_file', '') or '').strip()
    hod_zone = str(getattr(args, 'hod_zone', '') or '').strip()
    cosmo = normalize_cosmo_token(getattr(args, 'cosmo', ''))
    hod = str(getattr(args, 'hod', '') or '').strip()
    hod_root = Path(args.hod_root).expanduser()
    hod_run_dir = None

    if hod and not hod_zone and not raw_file_arg:
        hod_zone = build_hod_zone_label(cosmo=cosmo, hod=hod,
                                        phase=getattr(args, 'phase', 'ph000'),
                                        seed_token=getattr(args, 'seed_token', 'seed0'))

    if raw_file_arg:
        raw_file = resolve_existing_path(raw_file_arg)
        hod_run_dir = _run_dir_from_raw_path(raw_file)
        if not hod_zone:
            hod_zone = _zone_from_raw_path(raw_file)
        if not cosmo and hod_zone:
            cosmo = normalize_cosmo_token(_cosmo_from_zone(hod_zone))
    else:
        if hod_zone and not cosmo:
            cosmo = normalize_cosmo_token(_cosmo_from_zone(hod_zone))
        if not cosmo:
            raise ValueError('For --dataset hod, pass --cosmo, --hod-zone, or --raw-file.')
        existing = []
        for run_dir in _candidate_hod_run_dirs(hod_root, cosmo=cosmo, hod_zone=hod_zone):
            for path in _raw_files_in_run_dir(run_dir, hod_zone=hod_zone):
                existing.append(path)
        if not existing:
            searched = ', '.join(str(path / 'raw') for path in _candidate_hod_run_dirs(
                hod_root, cosmo=cosmo, hod_zone=hod_zone))
            raise FileNotFoundError()
        if len(existing) > 1:
            names = ', '.join(path.name for path in existing[:5])
            roots = ', '.join(str(path.parent.parent) for path in existing[:5])
            raise RuntimeError()
        raw_file = existing[0].resolve()
        hod_run_dir = _run_dir_from_raw_path(raw_file)
        if not hod_zone:
            hod_zone = _zone_from_raw_path(raw_file)

    if not hod_zone:
        raise ValueError(f'Could not infer HOD zone from raw file: {raw_file}')
    if not cosmo:
        cosmo = normalize_cosmo_token(_cosmo_from_zone(hod_zone))
    if hod_run_dir is None:
        hod_run_dir = _run_dir_from_raw_path(raw_file)

    user_prob = str(getattr(args, 'astra_prob_file', '') or '').strip()
    if user_prob and not _is_default_dr2_probability(user_prob):
        prob_file = resolve_existing_path(user_prob)
    else:
        prob_file = (hod_run_dir / 'release' / 'probabilities'
                     / f'zone_{hod_zone}_probability_iterdata.fits.gz').resolve()

    return str(raw_file), str(prob_file), hod_zone, cosmo


def resolve_hod_probability_file(args):
    _, prob_file, _, _ = resolve_hod_paths(args)
    return prob_file


def _fits_lookup(data):
    return {name.upper(): name for name in list(data.names or [])}


def _read_hod_slice(data, lookup, rows, columns, optional_columns=()):
    missing = [col for col in columns if col.upper() not in lookup]
    if missing:
        raise KeyError(f'Missing HOD raw columns: {missing}')
    out = {}
    for col in list(columns) + list(optional_columns):
        key = lookup.get(col.upper())
        if key is not None:
            out[col] = np.asarray(data[key][rows]).copy()
    return out


def _positions_from_cartesian_columns(columns, cartesian_cols, scale):
    xcol, ycol, zcol = cartesian_cols
    pos = np.column_stack([np.asarray(columns[xcol], dtype=np.float64),
                           np.asarray(columns[ycol], dtype=np.float64),
                           np.asarray(columns[zcol], dtype=np.float64)])
    if scale != 1.0:
        pos *= float(scale)
    return pos


def _finite_catalog_mask(pos, weights=None):
    mask = np.all(np.isfinite(pos), axis=1)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights >= 0.0)
    return mask


def _catalog_tag_dr2(args):
    region = str(getattr(args, 'sky_region', 'NGC') or 'NGC').upper()
    suffix = '' if region == 'NGC' else f'_{region}'
    return f'{args.tracer}{suffix}_z{args.zmin:.3f}_{args.zmax:.3f}_N{args.grid}'


def _catalog_tag_hod(args, zone):
    tracer = str(getattr(args, 'tracer', 'HOD') or 'HOD')
    return f'{tracer}_{zone}_N{args.grid}'


def prepare_catalog(args, need_targetid=True):
    dataset = str(getattr(args, 'dataset', 'dr2') or 'dr2').lower()
    if dataset == 'dr2':
        return _prepare_dr2_catalog(args, need_targetid=need_targetid)
    if dataset == 'hod':
        return _prepare_hod_catalog(args, need_targetid=need_targetid)
    raise ValueError()


def _prepare_dr2_catalog(args, need_targetid=True):
    position_source = str(getattr(args, 'position_source', 'auto') or 'auto').lower()
    if position_source == 'cartesian':
        raise ValueError('DR2 LSS catalogs are read from RA/DEC/Z')

    base_cols = ['RA', 'DEC', 'Z']
    real_required = base_cols + (['TARGETID'] if need_targetid else [])
    optional = ['WEIGHT']
    data_path = Path(args.base_dir) / f'{args.tracer}_clustering.dat.fits'
    if not data_path.is_file():
        raise FileNotFoundError(f'Data catalog not found: {data_path}')

    real = read_fits_columns_case_insensitive(str(data_path), real_required, optional_columns=optional)
    rand, random_indices_used, rand_paths_used = load_and_stack_randoms(base_dir=args.base_dir,
                                                                        tracer=args.tracer,
                                                                        start_index=args.random_index,
                                                                        n_random_files=args.n_random_files,
                                                                        columns=base_cols,
                                                                        zmin=args.zmin,
                                                                        zmax=args.zmax,
                                                                        optional_columns=optional)

    n_data_before_selection = len(real)
    n_random_before_selection = len(rand)

    md = (real['Z'] >= args.zmin) & (real['Z'] < args.zmax)
    mr = np.ones(len(rand), dtype=bool)
    region = str(getattr(args, 'sky_region', 'NGC') or 'NGC').upper()
    if region != 'ALL':
        md &= sky_region_mask(real['RA'], region)
        mr &= sky_region_mask(rand['RA'], region)
    real = real[md]
    rand = rand[mr]

    if len(real) == 0:
        raise RuntimeError('No data objects selected after DR2 cuts.')
    if len(rand) == 0:
        raise RuntimeError('No random objects selected after DR2 cuts.')

    if args.random_subsample < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.random(len(rand)) < args.random_subsample
        rand = rand[keep]

    if len(rand) == 0:
        raise RuntimeError('Random catalog is empty after random subsampling')

    cosmo = FlatLambdaCDM(H0=args.h0, Om0=args.om0)
    chi_d = np.asarray(cosmo.comoving_distance(real['Z']).value, dtype=np.float64)
    chi_r = np.asarray(cosmo.comoving_distance(rand['Z']).value, dtype=np.float64)
    pos_d = radec_to_cartesian(real['RA'], real['DEC'], chi_d)
    pos_r = radec_to_cartesian(rand['RA'], rand['DEC'], chi_r)
    w_d = get_weight_column(real)
    w_r = get_weight_column(rand)
    targetid_d = np.asarray(real['TARGETID'], dtype=np.int64) if need_targetid else None
    prob_file = resolve_dr2_probability_file(args) if hasattr(args, 'astra_prob_file') else ''

    return PreparedCatalog(
        dataset='dr2', tracer=args.tracer, zone=region,
        data_path=str(data_path),
        random_paths=rand_paths_used,
        random_indices_used=random_indices_used,
        astra_prob_file=prob_file,
        pos_d=pos_d, pos_r=pos_r,
        w_d=w_d, w_r=w_r,
        targetid_d=targetid_d,
        n_data=len(pos_d),
        n_random=len(pos_r),
        n_data_before_selection=n_data_before_selection,
        n_random_before_selection=n_random_before_selection,
        sky_region_desc=sky_region_description(region),
        tag=_catalog_tag_dr2(args),
        metadata={'input_family': 'dr2_lss',
                  'base_dir': str(args.base_dir),
                  'position_source': 'radec-z',
                  'sky_region': region})


def _prepare_hod_catalog(args, need_targetid=True):
    raw_file, prob_file, zone, cosmo = resolve_hod_paths(args)
    args.tracer = 'HOD'
    hod = _hod_from_zone(zone)
    phase = _phase_from_zone(zone)
    seed_token = _seed_token_from_zone(zone)
    cartesian_cols = list(getattr(args, 'cartesian_cols', ['XCART', 'YCART', 'ZCART']))
    if len(cartesian_cols) != 3:
        raise ValueError('--cartesian-cols must provide exactly 3 column names.')

    with fits.open(raw_file, memmap=True) as hdul:
        hdu = hdul[1]
        data = hdu.data
        if data is None:
            raise RuntimeError()
        header = hdu.header
        lookup = _fits_lookup(data)
        n_rows = int(header.get('NAXIS2', len(data)))
        n_data = int(header.get('NDATA', 0))
        n_random_available = int(header.get('NRAND', 0))
        if n_data <= 0:
            raise RuntimeError()
        if n_random_available <= 0:
            inferred = n_rows // n_data - 1
            n_random_available = max(0, inferred)
        if n_random_available <= 0:
            raise RuntimeError(f'HOD raw file has no random iterations: {raw_file}')

        random_indices = list(range(args.random_index, args.random_index + args.n_random_files))
        bad = [idx for idx in random_indices if idx < 0 or idx >= n_random_available]
        if bad:
            raise ValueError(f'Requested HOD random indices outside available range 0-{n_random_available - 1}: {bad}')

        required_data = ['TARGETID'] + cartesian_cols if need_targetid else list(cartesian_cols)
        optional = ['WEIGHT']
        data_cols = _read_hod_slice(data, lookup, slice(0, n_data), required_data, optional_columns=optional)
        pos_d = _positions_from_cartesian_columns(data_cols, cartesian_cols, args.cartesian_scale)
        w_d = (np.asarray(data_cols['WEIGHT'], dtype=np.float64)
               if 'WEIGHT' in data_cols else np.ones(n_data, dtype=np.float64))
        targetid_d = np.asarray(data_cols['TARGETID'], dtype=np.int64) if need_targetid else None

        random_pos_chunks = []
        random_weight_chunks = []
        for idx in random_indices:
            start = n_data * (idx + 1)
            stop = start + n_data
            rand_cols = _read_hod_slice(data, lookup, slice(start, stop), cartesian_cols, optional_columns=optional)
            random_pos_chunks.append(_positions_from_cartesian_columns(rand_cols, cartesian_cols, args.cartesian_scale))
            if 'WEIGHT' in rand_cols:
                random_weight_chunks.append(np.asarray(rand_cols['WEIGHT'], dtype=np.float64))

        boxsize = header.get('HODBOX', header.get('LBOX', header.get('BOXSZX', None)))
        box_min = header.get('BOXMIN', None)
        box_max = header.get('BOXMAX', None)

    pos_r = np.concatenate(random_pos_chunks, axis=0)
    if random_weight_chunks:
        w_r = np.concatenate(random_weight_chunks).astype(np.float64, copy=False)
    else:
        w_r = np.ones(pos_r.shape[0], dtype=np.float64)

    mask_d = _finite_catalog_mask(pos_d, w_d)
    mask_r = _finite_catalog_mask(pos_r, w_r)
    if targetid_d is not None:
        targetid_d = targetid_d[mask_d]
    pos_d = pos_d[mask_d]
    w_d = np.nan_to_num(w_d[mask_d], nan=0.0, posinf=0.0, neginf=0.0)
    pos_r = pos_r[mask_r]
    w_r = np.nan_to_num(w_r[mask_r], nan=0.0, posinf=0.0, neginf=0.0)

    if args.random_subsample < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.random(pos_r.shape[0]) < args.random_subsample
        pos_r = pos_r[keep]
        w_r = w_r[keep]

    if pos_d.shape[0] == 0:
        raise RuntimeError()
    if pos_r.shape[0] == 0:
        raise RuntimeError()

    boxsize_default = float(boxsize) * float(args.cartesian_scale) if boxsize is not None else None
    box_origin_default = None
    box_center_default = None
    if box_min is not None and box_max is not None:
        box_min = float(box_min) * float(args.cartesian_scale)
        box_max = float(box_max) * float(args.cartesian_scale)
        box_origin_default = np.array([box_min, box_min, box_min], dtype=np.float64)
        box_center_default = np.array([0.5 * (box_min + box_max)] * 3, dtype=np.float64)
        if boxsize_default is None:
            boxsize_default = float(box_max - box_min)

    random_paths = [f'{raw_file}[RANDITER={idx}]' for idx in random_indices]
    return PreparedCatalog(
        dataset='hod', tracer='HOD', zone=zone,
        data_path=raw_file,
        random_paths=random_paths,
        random_indices_used=random_indices,
        astra_prob_file=str(prob_file),
        pos_d=pos_d, pos_r=pos_r,
        w_d=w_d, w_r=w_r,
        targetid_d=targetid_d,
        n_data=pos_d.shape[0],
        n_random=pos_r.shape[0],
        n_data_before_selection=n_data,
        n_random_before_selection=n_data * len(random_indices),
        sky_region_desc='HOD full box',
        tag=_catalog_tag_hod(args, zone),
        boxsize_default=boxsize_default,
        box_origin_default=box_origin_default,
        box_center_default=box_center_default,
        metadata={'input_family': 'hod_astra_fullbox',
                  'hod_root': str(args.hod_root),
                  'hod_run_dir': str(_run_dir_from_raw_path(raw_file)),
                  'cosmo': cosmo,
                  'hod': hod,
                  'phase': phase,
                  'seed_token': seed_token,
                  'zone': zone,
                  'position_source': 'cartesian',
                  'cartesian_cols': cartesian_cols,
                  'cartesian_scale': float(args.cartesian_scale),
                  'n_random_available': int(n_random_available)})


def compute_box_geometry(pos_d, pos_r, boxsize_arg, box_padding):
    mins = np.minimum(np.min(pos_d, axis=0), np.min(pos_r, axis=0)).astype(np.float64)
    maxs = np.maximum(np.max(pos_d, axis=0), np.max(pos_r, axis=0)).astype(np.float64)
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
    shifted = (positions.astype(np.float64) - origin).astype(np.float64)
    upper = np.nextafter(np.float64(boxsize), np.float64(0.0))
    np.clip(shifted, 0.0, upper, out=shifted)
    return shifted


def apply_box_geometry(pos_d, pos_r, args, catalog=None):
    use_catalog_box = False
    if catalog is not None and catalog.boxsize_default is not None and catalog.box_origin_default is not None:
        requested = float(getattr(args, 'boxsize', 0.0) or 0.0)
        use_catalog_box = requested <= 0.0 or np.isclose(requested, catalog.boxsize_default)

    if use_catalog_box:
        boxsize = float(catalog.boxsize_default)
        origin = np.asarray(catalog.box_origin_default, dtype=np.float64)
        center = np.asarray(catalog.box_center_default, dtype=np.float64)
    else:
        boxsize, center, origin = compute_box_geometry(pos_d, pos_r,
                                                       boxsize_arg=float(getattr(args, 'boxsize', 0.0) or 0.0),
                                                       box_padding=float(getattr(args, 'box_padding', 0.0) or 0.0))

    pos_d_box = shift_and_clip_to_box(pos_d, origin=origin, boxsize=boxsize)
    pos_r_box = shift_and_clip_to_box(pos_r, origin=origin, boxsize=boxsize)
    boxcenter = np.array([0.5 * boxsize] * 3, dtype=np.float64)
    return pos_d_box, pos_r_box, boxsize, center, origin, boxcenter


def apply_box_geometry_single(positions, args, catalog=None):
    if catalog is not None and catalog.boxsize_default is not None and catalog.box_origin_default is not None:
        requested = float(getattr(args, 'boxsize', 0.0) or 0.0)
        if requested <= 0.0 or np.isclose(requested, catalog.boxsize_default):
            boxsize = float(catalog.boxsize_default)
            origin = np.asarray(catalog.box_origin_default, dtype=np.float64)
            center = np.asarray(catalog.box_center_default, dtype=np.float64)
            shifted = shift_and_clip_to_box(positions, origin=origin, boxsize=boxsize)
            boxcenter = np.array([0.5 * boxsize] * 3, dtype=np.float64)
            return shifted, boxsize, center, origin, boxcenter

    mins = np.min(positions, axis=0).astype(np.float64)
    maxs = np.max(positions, axis=0).astype(np.float64)
    lengths = maxs - mins
    requested = float(getattr(args, 'boxsize', 0.0) or 0.0)
    if requested > 0.0:
        boxsize = requested
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize
    else:
        boxsize = float(np.max(lengths) + 2.0 * float(getattr(args, 'box_padding', 0.0) or 0.0))
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize
    shifted = shift_and_clip_to_box(positions, origin=origin, boxsize=boxsize)
    boxcenter = np.array([0.5 * boxsize] * 3, dtype=np.float64)
    return shifted, boxsize, center, origin, boxcenter


def print_catalog_summary(catalog, outdir, args, resampler):
    print(f'---> dataset:         {catalog.dataset}')
    print(f'---> data catalog:    {catalog.data_path}')
    print('---> random catalogs: ' + ', '.join(str(p) for p in catalog.random_paths[:5])
          + (' ...' if len(catalog.random_paths) > 5 else ''))
    print(f'---> random stacking: start={args.random_index}, '
          f'n_files={args.n_random_files}, indices={catalog.random_indices_used}')
    if catalog.astra_prob_file:
        print(f'---> astra probs:     {catalog.astra_prob_file}')
    print(f'---> sky/box:         {catalog.sky_region_desc}')
    print(f'---> selected data objects:   {catalog.n_data} (from {catalog.n_data_before_selection})')
    print(f'---> selected random objects: {catalog.n_random} (from {catalog.n_random_before_selection})')
    if args.random_subsample < 1.0:
        print(f'---> random subsample fraction={args.random_subsample:.3f}')
    print(f'---> outdir:          {outdir}')
    print(f'---> nmesh:           {args.grid}')
    print(f'---> mas/resampler:   {args.mas}/{resampler}')
    print(f'---> interlacing:     {args.interlacing}')
    print(f'---> nthreads hint:   {args.nthreads}')