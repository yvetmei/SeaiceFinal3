from xml.parsers.expat import model

import numpy as np
import matplotlib.pyplot as plt
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')                    
import re
import sys
import glob
import logging
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from tensorflow.keras.layers import LSTM, Dense, Dropout


try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[警告] rasterio 未安装，TIF 采样功能将不可用。pip install rasterio")

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("[警告] xarray 未安装，NC 读取功能将不可用。pip install xarray netcdf4")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[警告] h5py 未安装，H5 深度文件读取将不可用。pip install h5py")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("[警告] pyproj 未安装，极地投影坐标转换将不可用。pip install pyproj")

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    class MinMaxScaler:
        """MinMaxScaler 轻量回退实现"""
        def __init__(self):
            self.min_ = None
            self.scale_ = None

        def fit(self, X):
            arr = np.array(X, dtype=float)
            self.min_ = np.nanmin(arr, axis=0)
            mx = np.nanmax(arr, axis=0)
            self.scale_ = mx - self.min_
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            return (np.array(X, dtype=float) - self.min_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, X):
            return np.array(X, dtype=float) * self.scale_ + self.min_

# 一、多源异构数据空间采样与时序对齐模块
def sample_tif(tif_path: str, lon: float, lat: float) -> float:
    """
    从 .tif 文件中按经纬度采样单点值。
    自动检测 CRS，若非 WGS84 则通过 pyproj 转换为投影坐标后采样。
    """
    if not HAS_RASTERIO or not os.path.exists(tif_path):
        return np.nan
    try:
        with rasterio.open(tif_path) as src:
            crs = src.crs
            if crs and crs.to_epsg() != 4326 and HAS_PYPROJ:
                transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat

            vals = list(src.sample([(x, y)]))
            val  = float(vals[0][0])

            if src.nodata is not None and np.isclose(val, src.nodata):
                return np.nan
            return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def sample_nc_monthly(nc_path: str, lon: float, lat: float,
                      year: int, month: int, var_name: str = 'h') -> float:

    if not HAS_XARRAY or not os.path.exists(nc_path):
        return np.nan
    try:
        with xr.open_dataset(nc_path, decode_times=False) as ds:
            if var_name not in ds:
                return np.nan

            da = ds[var_name]
            if 'time' in da.dims and da.shape[da.dims.index('time')] == 12:
                da = da.isel(time=month - 1)

            lat_keys = [k for k in list(ds.coords) + list(ds.data_vars) if 'lat' in k.lower()]
            lon_keys = [k for k in list(ds.coords) + list(ds.data_vars) if 'lon' in k.lower()]
            if not lat_keys or not lon_keys:
                return np.nan

            lats = ds[lat_keys[0]].values
            lons = ds[lon_keys[0]].values

            if lats.ndim == 1:
                lat_idx = int(np.argmin(np.abs(lats - lat)))
                lon_idx = int(np.argmin(np.abs(lons - lon)))
                val     = float(da.values[lat_idx, lon_idx])
            else:
                dist    = (lats - lat) ** 2 + (lons - lon) ** 2
                lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)
                val     = float(da.values[lat_idx, lon_idx])

            return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def sample_h5_depth(h5_path: str, lon: float, lat: float,
                    var_name: str = 'depth') -> float:
    """
    从静态地形深度 .h5 文件中进行最近邻点匹配采样。
    """
    if not HAS_H5PY or not os.path.exists(h5_path):
        return np.nan
    try:
        with h5py.File(h5_path, 'r') as f:
            depth_matrix = f[var_name][:]
            h5_lons = f['lon'][:]
            h5_lats = f['lat'][:]

            if h5_lons.ndim == 1:
                lon_idx = int(np.argmin(np.abs(h5_lons - lon)))
                lat_idx = int(np.argmin(np.abs(h5_lats - lat)))
                val     = float(depth_matrix[lat_idx, lon_idx])
            else:
                dist    = (h5_lats - lat) ** 2 + (h5_lons - lon) ** 2
                lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)
                val     = float(depth_matrix[lat_idx, lon_idx])

            return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


SIC_PERCENT_TO_FRAC = 0.01   # nc 密集度百分数 -> 0-1

def _match_col(cols, candidates, prefixes):
    """
    在列名字典 cols(小写->原名) 中匹配列。
    先精确匹配 candidates, 再按 prefixes 前缀匹配
    """
    for k in candidates:
        if k in cols:
            return cols[k]
    for low, orig in cols.items():
        for p in prefixes:
            if low.startswith(p) and 'speed' not in low and 'dir' not in low:
                return orig
    return None


def _load_buoy_files(excel_dir, doy_base_year=2018):
    """
    读取 excel_data 下所有浮标 xlsx, 合并为按时间排序的 DataFrame
    """
    files = sorted(glob.glob(os.path.join(excel_dir, '*.xlsx')))
    if not files:
        raise FileNotFoundError(f'{excel_dir} 下没有 xlsx 文件')
    frames = []
    n_skip = 0
    for fp in files:
        buoy_id = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logging.warning(f'读取 {fp} 失败: {e}')
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        lat_c = _match_col(cols, ('lat', 'latitude'),          ('lat',))
        lon_c = _match_col(cols, ('long', 'lon', 'longitude'), ('lon', 'long'))

        time_c = None       
        doy_c  = None       
        for k in ('time', 'datetime', 'date'):
            if k in cols:
                time_c = cols[k]; break
        if time_c is None:
            for c in df.columns:
                if str(c).lower().startswith('unnamed'):
                    p = pd.to_datetime(df[c], errors='coerce')
                    if p.notna().mean() > 0.8 and (p.dropna().dt.year >= 1990).mean() > 0.5:
                        time_c = c; break
        if time_c is None:
            for c in df.columns:
                if c in (lat_c, lon_c):
                    continue
                low = str(c).lower()
                num = pd.to_numeric(df[c], errors='coerce')
                if (low.startswith('time') or 'doy' in low or 'day' in low) \
                        and num.notna().mean() > 0.8 and num.dropna().between(1, 550).all():
                    if doy_c is None:
                        doy_c = c
                    continue
                p = pd.to_datetime(df[c], errors='coerce')
                if p.notna().mean() > 0.8 and (p.dropna().dt.year >= 1990).mean() > 0.5:
                    time_c = c; break

        if not (lat_c and lon_c and (time_c is not None or doy_c is not None)):
            logging.warning(f'{buoy_id}: 找不到 time/lat/lon 列, 跳过。列={list(df.columns)}')
            n_skip += 1
            continue

        if time_c is not None:
            parsed_time = pd.to_datetime(df[time_c], errors='coerce')
            if (parsed_time.dropna().dt.year < 1990).mean() > 0.5 and doy_c is not None:
                time_c = None  
        if time_c is None:
            num = pd.to_numeric(df[doy_c], errors='coerce')
            base = pd.Timestamp(doy_base_year, 1, 1)
            parsed_time = base + pd.to_timedelta(num - 1.0, unit='D')
            logging.info(f'  {buoy_id}: time 列 "{doy_c}" 为 day-of-year, '
                         f'已按基准年 {doy_base_year} 还原')

        data = {
            'time': parsed_time,
            'lat':  pd.to_numeric(df[lat_c], errors='coerce'),
            'lon':  pd.to_numeric(df[lon_c], errors='coerce'),
        }
        mconc_c = _match_col(cols, ('m_conc', 'conc', 'concentration', 'sic'), ('m_conc',))
        if mconc_c:
            data['buoy_A'] = pd.to_numeric(df[mconc_c], errors='coerce') * 0.01  
        iceu_c = _match_col(cols, ('ice_u', 'u', 'u_ice'), ('ice_u',))
        icev_c = _match_col(cols, ('ice_v', 'v', 'v_ice'), ('ice_v',))
        if iceu_c and icev_c:
            data['ice_u'] = pd.to_numeric(df[iceu_c], errors='coerce')
            data['ice_v'] = pd.to_numeric(df[icev_c], errors='coerce')

        sub = pd.DataFrame(data)
        sub['buoy_id'] = buoy_id
        sub = sub.dropna(subset=['time', 'lat', 'lon'])
        frames.append(sub)
        extra = ' (含ice_u/ice_v)' if 'ice_u' in sub.columns else ''
        logging.info(f'  浮标 {buoy_id}: {len(sub)} 点 '
                     f'[{sub.time.min()} ~ {sub.time.max()}]{extra}')
    if not frames:
        raise RuntimeError('没有任何浮标文件读取成功')
    allbuoy = pd.concat(frames, ignore_index=True).sort_values('time').reset_index(drop=True)
    logging.info(f'合并浮标点总数: {len(allbuoy)}, 浮标数: {allbuoy.buoy_id.nunique()}'
                 f'{f", 跳过 {n_skip} 个" if n_skip else ""}')
    return allbuoy


class _TifVelocitySampler:
    """
    按日期采样 u/v tif (EPSG:3413)。
      {YYYYMMDD}v_uv.tif  -> u 分量 (东西向)
      {YYYYMMDD}v_vv.tif  -> v 分量 (南北向)
    """
    def __init__(self, tif_dir):
        # u/v 都在 v 子目录; 若无 v 子目录则退回 tif_dir 本身
        cand = os.path.join(tif_dir, 'v')
        self.uv_dir = cand if os.path.isdir(cand) else tif_dir
        if not HAS_PYPROJ:
            raise ImportError('需要 pyproj: pip install pyproj')
        if not HAS_RASTERIO:
            raise ImportError('需要 rasterio: pip install rasterio')
        from pyproj import Transformer
        self._tf = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
        self._cache = {}

    def _load_day(self, date_str):
        if date_str in self._cache:
            return self._cache[date_str]
        # _uv 后缀 = u 分量, _vv 后缀 = v 分量
        up = os.path.join(self.uv_dir, f'{date_str}v_uv.tif')
        vp = os.path.join(self.uv_dir, f'{date_str}v_vv.tif')
        if not (os.path.exists(up) and os.path.exists(vp)):
            self._cache[date_str] = None
            return None
        with rasterio.open(up) as su, rasterio.open(vp) as sv:
            self._cache[date_str] = (su.read(1), sv.read(1),
                                     su.index, su.height, su.width, su.nodata)
        return self._cache[date_str]

    def sample(self, lat, lon, date):
        day = self._load_day(date.strftime('%Y%m%d'))
        if day is None:
            return np.nan, np.nan
        ua, va, index_fn, H, W, nodata = day
        x, y = self._tf.transform(lon, lat)
        try:
            row, col = index_fn(x, y)
        except Exception:
            return np.nan, np.nan
        if not (0 <= row < H and 0 <= col < W):
            return np.nan, np.nan
        u = float(ua[row, col]); v = float(va[row, col])
        if nodata is not None and (u == nodata or v == nodata):
            return np.nan, np.nan
        # 单位统一: tif 原始为 cm/s -> m/s (与浮标 ice_u/v 及 OSI SAF 弱标签一致)
        return u * 0.01, v * 0.01


class _NcThicknessSampler:
    """采样 SIT_25km_monthly_YYYYMM.nc 的 A/h, 含空间 IDW + 跨月时间线性插值,
    消除月度场作为逐时刻输入时的阶梯常数效应。"""
    def __init__(self, nc_dir, nc_pattern='SIT_25km_monthly_{ym}.nc',
                 spatial='idw', temporal=True, k_neighbors=4):
        if not HAS_XARRAY:
            raise ImportError('需要 xarray: pip install xarray netcdf4')
        self.nc_dir = nc_dir
        self.pattern = nc_pattern
        self.spatial = spatial
        self.temporal = bool(temporal)
        self.k_neighbors = int(k_neighbors)
        self._cache = {}

    def _load_month(self, ym):
        if ym in self._cache:
            return self._cache[ym]
        fp = os.path.join(self.nc_dir, self.pattern.format(ym=ym))
        if not os.path.exists(fp):
            yyyy, mm = ym[:4], ym[4:6]
            cand = []
            for f in os.listdir(self.nc_dir):
                if not f.endswith('.nc'):
                    continue
                if ym in f:                       
                    cand.append(f)
                elif yyyy in f and (f'_{mm}' in f or f'-{mm}' in f
                                    or f'{mm}.' in f.split(yyyy)[-1]):
                    cand.append(f)
            if not cand:
                if not getattr(self, '_warned_missing', False):
                    try:
                        examples = [f for f in os.listdir(self.nc_dir)
                                    if f.endswith('.nc')][:3]
                    except Exception:
                        examples = []
                    logging.warning(f'[nc] 找不到 {ym} 对应的 nc。目录 {self.nc_dir} '
                                    f'中的文件名示例: {examples}。'
                                    f'若格式不同请用 --nc_pattern 指定。')
                    self._warned_missing = True
                self._cache[ym] = None
                return None
            fp = os.path.join(self.nc_dir, sorted(cand)[0])
        ds = xr.open_dataset(fp, decode_times=False)
        lat2d = ds['lat'].values
        lon2d = ds['lon'].values
        h2d = np.asarray(ds['sea_ice_thickness'].values).squeeze()
        A2d = np.asarray(ds['sea_ice_concentration'].values).squeeze() * SIC_PERCENT_TO_FRAC
        ds.close()
        self._cache[ym] = (lat2d, lon2d, h2d, A2d)
        return self._cache[ym]

    def _sample_month(self, m, lat, lon):
        if m is None:
            return np.nan, np.nan
        lat2d, lon2d, h2d, A2d = m
        d2 = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        if self.spatial == 'nearest':
            i, j = np.unravel_index(np.argmin(d2), d2.shape)
            A = float(A2d[i, j]); h = float(h2d[i, j])
            return (A if np.isfinite(A) else np.nan,
                    h if np.isfinite(h) else np.nan)
        k = max(1, min(self.k_neighbors, d2.size))
        flat = d2.ravel()
        idx = np.argpartition(flat, k - 1)[:k]
        Af = A2d.ravel()[idx]; hf = h2d.ravel()[idx]; df = flat[idx]
        zero = df <= 1e-12
        if np.any(zero):
            A0 = Af[zero]; h0 = hf[zero]
            A = float(A0[np.isfinite(A0)][0]) if np.any(np.isfinite(A0)) else np.nan
            h = float(h0[np.isfinite(h0)][0]) if np.any(np.isfinite(h0)) else np.nan
            return A, h
        w = 1.0 / df
        mA = np.isfinite(Af); mh = np.isfinite(hf)
        A = float(np.sum(w[mA] * Af[mA]) / np.sum(w[mA])) if np.any(mA) else np.nan
        h = float(np.sum(w[mh] * hf[mh]) / np.sum(w[mh])) if np.any(mh) else np.nan
        return A, h

    @staticmethod
    def _month_anchor(year, month):
        return datetime(year, month, 15)

    def _bracket_months(self, date):
        y, mo = date.year, date.month
        anchor = self._month_anchor(y, mo)
        if date >= anchor:
            prev = (y, mo)
            nxt = (y + 1, 1) if mo == 12 else (y, mo + 1)
        else:
            nxt = (y, mo)
            prev = (y - 1, 12) if mo == 1 else (y, mo - 1)
        return prev, nxt

    def sample(self, lat, lon, date):
        if not self.temporal:
            m = self._load_month(date.strftime('%Y%m'))
            return self._sample_month(m, lat, lon)
        (py, pm), (ny, nm) = self._bracket_months(date)
        t_prev = self._month_anchor(py, pm); t_next = self._month_anchor(ny, nm)
        m_prev = self._load_month(f'{py:04d}{pm:02d}')
        m_next = self._load_month(f'{ny:04d}{nm:02d}')
        A_p, h_p = self._sample_month(m_prev, lat, lon)
        A_n, h_n = self._sample_month(m_next, lat, lon)
        span = (t_next - t_prev).total_seconds()
        w = 0.0 if span <= 0 else (date - t_prev).total_seconds() / span
        w = min(1.0, max(0.0, w))
        def _interp(a, b):
            if not np.isfinite(a) and not np.isfinite(b): return np.nan
            if not np.isfinite(a): return b
            if not np.isfinite(b): return a
            return (1.0 - w) * a + w * b
        return _interp(A_p, A_n), _interp(h_p, h_n)


def build_dataset_from_dataForIce(base_dir, nc_dir=None,
                                  nc_pattern='SIT_25km_monthly_{ym}.nc',
                                  resample_rule='1D', min_valid_ratio=1.0,
                                  max_gap_days=3, speed_jump_mps=0.5):
    excel_dir = os.path.join(base_dir, 'excel_data')
    tif_dir   = os.path.join(base_dir, 'tif_data')
    if nc_dir is None:
        nc_dir = os.path.join(base_dir, 'nc_data')

    logging.info('=== [dataForIce] 读取浮标轨迹 ===')
    buoy = _load_buoy_files(excel_dir)

    logging.info(f'=== [dataForIce] 轨迹按 {resample_rule} 重采样 ===')
    has_ice_uv = ('ice_u' in buoy.columns) and ('ice_v' in buoy.columns)
    has_buoy_A = 'buoy_A' in buoy.columns
    agg_cols = (['lat', 'lon']
                + (['ice_u', 'ice_v'] if has_ice_uv else [])
                + (['buoy_A'] if has_buoy_A else []))
    parts = []
    for bid, g in buoy.groupby('buoy_id'):
        g = g.set_index('time').sort_index()
        gr = g[agg_cols].resample(resample_rule).mean().dropna(subset=['lat', 'lon'])
        gr['buoy_id'] = bid
        parts.append(gr.reset_index())
    traj = pd.concat(parts, ignore_index=True)
    # 关键: 按 (浮标, 时间) 排序, 而非全局只按 time 混排
    traj = traj.sort_values(['buoy_id', 'time']).reset_index(drop=True)
    logging.info(f'重采样后轨迹点: {len(traj)} (浮标 {traj["buoy_id"].nunique()} 个)')

    vel = _TifVelocitySampler(tif_dir)
    ncs = _NcThicknessSampler(nc_dir, nc_pattern=nc_pattern)

    logging.info('=== [dataForIce] 逐点采样 [u,v,A,h] ===')
    feats, dates, n_drop, n_selfuv = [], [], 0, 0
    row_bid, row_time = [], []   # 与 feats 行对齐, 供切段
    n_uv_ok = n_A_ok = n_h_ok = 0
    for _, row in traj.iterrows():
        t = row['time'].to_pydatetime()
        lat, lon = float(row['lat']), float(row['lon'])
        # u/v: 优先用浮标自带 ice_u/ice_v, 缺失才从 tif 采样
        u = v = np.nan
        if has_ice_uv and np.isfinite(row.get('ice_u', np.nan)) \
                and np.isfinite(row.get('ice_v', np.nan)):
            u, v = float(row['ice_u']), float(row['ice_v'])
            n_selfuv += 1
        else:
            u, v = vel.sample(lat, lon, t)
        # A/h: nc 月度厚度为主; 浓度优先用浮标逐点 m_conc(时间分辨率更高)
        A, h = ncs.sample(lat, lon, t)
        if has_buoy_A and np.isfinite(row.get('buoy_A', np.nan)):
            A = float(row['buoy_A'])
        if np.isfinite(u) and np.isfinite(v): n_uv_ok += 1
        if np.isfinite(A): n_A_ok += 1
        if np.isfinite(h): n_h_ok += 1
        vals = [u, v, A, h]
        if np.mean([np.isfinite(x) for x in vals]) < min_valid_ratio:
            n_drop += 1
            continue
        feats.append([x if np.isfinite(x) else 0.0 for x in vals])
        dates.append(t.strftime('%Y-%m-%d'))
        row_bid.append(row['buoy_id'])
        row_time.append(row['time'])

    data = np.array(feats, dtype=float)
    N = max(len(traj), 1)
    logging.info(f'[dataForIce] 采样有效率: u/v={n_uv_ok}/{len(traj)} ({100*n_uv_ok/N:.0f}%), '
                 f'A={n_A_ok}/{len(traj)} ({100*n_A_ok/N:.0f}%), '
                 f'h={n_h_ok}/{len(traj)} ({100*n_h_ok/N:.0f}%)')
    if n_A_ok < 0.1 * N or n_h_ok < 0.1 * N:
        logging.warning('[dataForIce] *** A 或 h 有效率极低! 多半是 nc 文件名/年月没匹配上, '
                        '导致 A/h 被填 0。请检查 --nc_dir 路径和 --nc_pattern 文件名模板。***')

    # ---- 拉格朗日切段: 同浮标 + 时间连续 + 无速度突变 才算一段 ----
    seg_ids = np.full(len(data), -1, dtype=int)
    if len(data):
        row_time = pd.to_datetime(pd.Series(row_time)).reset_index(drop=True)
        seg = 0
        seg_ids[0] = 0
        for i in range(1, len(data)):
            same_buoy = (row_bid[i] == row_bid[i - 1])
            gap_days = (row_time[i] - row_time[i - 1]).total_seconds() / 86400.0
            du = data[i, 0] - data[i - 1, 0]
            dv = data[i, 1] - data[i - 1, 1]
            jump = float(np.hypot(du, dv))
            if (not same_buoy) or (gap_days > max_gap_days) or (jump > speed_jump_mps):
                seg += 1
            seg_ids[i] = seg
        n_seg = len(np.unique(seg_ids))
        seg_lens = np.bincount(seg_ids)
        logging.info(f'[dataForIce] 拉格朗日切段: {n_seg} 段, '
                     f'段长 min/中位/max = {seg_lens.min()}/{int(np.median(seg_lens))}/{seg_lens.max()} '
                     f'(gap>{max_gap_days}d 或 速度跳变>{speed_jump_mps}m/s 处切断)')

    logging.info(f'[dataForIce] 最终特征: {data.shape} (丢弃 {n_drop}, '
                 f'其中 {n_selfuv} 点用浮标自带 ice_u/ice_v)')
    if len(data):
        logging.info(f'  u:{data[:,0].min():.3f}~{data[:,0].max():.3f} '
                     f'v:{data[:,1].min():.3f}~{data[:,1].max():.3f} '
                     f'A:{data[:,2].min():.3f}~{data[:,2].max():.3f} '
                     f'h:{data[:,3].min():.3f}~{data[:,3].max():.3f}')
    return data, dates, seg_ids


def build_dataset_from_mixed_sources(excel_path: str, tif_dir: str,
                                     nc_dir: str, h5_path: str):

    print(f"\n[数据集成] 读取 Excel 轨迹坐标表: {excel_path}")
    df = pd.read_excel(excel_path, header=None)
    df.columns = ['DateTime', 'ID_or_Seq', 'Lat', 'Lon'] + list(df.columns[4:])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Year']     = df['DateTime'].dt.year
    df['Month']    = df['DateTime'].dt.month
    df['DateStr']  = df['DateTime'].dt.strftime('%Y%m%d')
    print(f"  成功加载 {len(df)} 条记录，开始多源时空交叉采样...")

    feats, dates = [], []

    for _, row in df.iterrows():
        lon      = float(row['Lon'])
        lat      = float(row['Lat'])
        date_str = row['DateStr']
        year     = int(row['Year'])
        month    = int(row['Month'])

        u_val     = sample_tif(os.path.join(tif_dir, 'u', f'u_{date_str}.tif'), lon, lat)
        v_val     = sample_tif(os.path.join(tif_dir, 'v', f'v_{date_str}.tif'), lon, lat)
        A_val     = sample_tif(os.path.join(tif_dir, 'A', f'A_{date_str}.tif'), lon, lat)
        h_val     = sample_nc_monthly(os.path.join(nc_dir, f'thickness_{year}.nc'),
                                      lon, lat, year, month, var_name='h')
        depth_val = sample_h5_depth(h5_path, lon, lat, var_name='depth')

        feats.append([u_val, v_val, A_val, h_val, depth_val])
        dates.append(row['DateTime'].strftime('%Y-%m-%d %H:%M'))

    feats = np.array(feats, dtype=float)
    dates = np.array(dates)

    df_feat = pd.DataFrame(feats, columns=['u', 'v', 'A', 'h', 'depth'])
    missing = df_feat.isna().sum().sum()
    if missing > 0:
        print(f"  [清洗] 发现 {missing} 个 NaN，执行时序线性插值...")
        df_feat = df_feat.interpolate(method='linear', limit_direction='both')

    final_feats = df_feat.values
    valid_mask  = np.isfinite(final_feats).all(axis=1)
    print(f"  [集成完毕] 输出 {np.sum(valid_mask)} 个完全对齐的 5 维时序样本。")
    return final_feats[valid_mask], dates[valid_mask]

def _parse_date_from_filename(fn: str):
    m = re.search(r'(\d{8})', fn)
    return m.group(1) if m else None


def _safe_open_dataset(path: str):
    """多引擎兼容的 xarray 数据集打开函数"""
    if not HAS_XARRAY:
        return None
    for engine in ['h5netcdf', 'netcdf4', None]:
        try:
            kwargs = {'decode_times': False}
            if engine:
                kwargs['engine'] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception:
            continue
    print(f'[警告] 无法打开文件: {path}，已跳过。')
    return None


def build_dataset_from_nc_dir(nc_dir: str, days: int = 730,
                               target_lat=None, target_lon=None):

    files = sorted(f for f in os.listdir(nc_dir) if f.lower().endswith('.nc'))
    if not files:
        raise FileNotFoundError(f'在 {os.path.abspath(nc_dir)} 中未找到 .nc 文件')

    n         = min(len(files), int(days)) if days and days > 0 else len(files)
    filepaths = [os.path.join(nc_dir, f) for f in files[-n:]]

    sample_ds = _safe_open_dataset(filepaths[0])
    if sample_ds is None:
        raise RuntimeError(f'无法打开文件: {filepaths[0]}')
    varnames = list(sample_ds.data_vars.keys())
    sample_ds.close()

    HAS_UVAH = all(v in varnames for v in ('u', 'v', 'A', 'h'))

    # --- 指定点采样 ---
    if target_lat is not None and target_lon is not None and HAS_UVAH:
        print(f"\n正在提取 lon={target_lon}, lat={target_lat} 处的数据...")
        ds0 = _safe_open_dataset(filepaths[0])
        lat_idx = lon_idx = None
        if ds0 is not None:
            for lat_name in ['lat', 'latitude', 'y', 'Y']:
                for lon_name in ['lon', 'longitude', 'x', 'X']:
                    if lat_name in ds0.data_vars and lon_name in ds0.data_vars:
                        lats = ds0[lat_name].values
                        lons = ds0[lon_name].values
                        lat_idx = int(np.argmin(np.abs(lats.ravel() - target_lat)))
                        lon_idx = int(np.argmin(np.abs(lons.ravel() - target_lon)))
                        break
                if lat_idx is not None:
                    break
            ds0.close()

        if lat_idx is not None:
            feats, dates = [], []
            for fp in filepaths:
                ds = _safe_open_dataset(fp)
                if ds is None:
                    continue
                try:
                    row = []
                    for vn in ['u', 'v', 'A', 'h']:
                        arr = ds[vn].values
                        val = (arr[-1, lat_idx, lon_idx] if arr.ndim == 3
                               else arr[lat_idx, lon_idx] if arr.ndim == 2
                               else np.nan)
                        row.append(float(val) if np.isfinite(val) else np.nan)
                    if all(np.isfinite(row)):
                        feats.append(row)
                        dates.append(_parse_date_from_filename(fp) or fp)
                except Exception:
                    pass
                finally:
                    ds.close()
            if feats:
                print(f"  成功提取 {len(feats)} 个时间点的数据。")
                return np.vstack(feats), np.array(dates)

    if HAS_UVAH:
        feats, dates = [], []
        for fp in filepaths:
            ds = _safe_open_dataset(fp)
            if ds is None:
                continue
            try:
                row = [np.nanmean(ds[v].values) for v in ['u', 'v', 'A', 'h']]
                feats.append(row)
                dates.append(_parse_date_from_filename(fp) or fp)
            except Exception:
                pass
            finally:
                ds.close()
        feats = np.vstack(feats)
        valid = np.isfinite(feats).all(axis=1)
        return feats[valid], np.array(dates)[valid]

    print('[信息] 未找到 u/v/A/h 字段，从可用标量场生成伪标签...')
    SNOW_CANDS = ('Snow_Depth', 'SnowDepth', 'snow_depth')
    arrays, dates = [], []
    for fp in filepaths:
        ds = _safe_open_dataset(fp)
        if ds is None:
            arrays.append(None)
        else:
            sv = next((c for c in SNOW_CANDS if c in ds.data_vars), None)
            if sv is None:
                sv = next((n for n, v in ds.data_vars.items()
                           if len(v.dims) == 2), None)
            arrays.append(ds[sv].values.astype(float) if sv else None)
            ds.close()
        dates.append(_parse_date_from_filename(fp) or fp)

    centroids, A_list, h_list = [], [], []
    for arr in arrays:
        if arr is None or not np.isfinite(arr).any():
            centroids.append((np.nan, np.nan))
            A_list.append(np.nan)
            h_list.append(np.nan)
            continue
        thr  = np.nanmax(arr) * 0.2
        mask = np.isfinite(arr) & (arr >= thr)
        if np.any(mask):
            idx = np.argwhere(mask)
            c   = idx.mean(axis=0)
            centroids.append((float(c[1]), float(c[0])))
        else:
            centroids.append((np.nan, np.nan))
        A_list.append(float(np.nanmean((arr >= np.nanmax(arr) * 0.05).astype(float))))
        h_list.append(float(np.nanmean(arr) * 0.1))

    u = np.full(len(centroids), np.nan)
    v = np.full(len(centroids), np.nan)
    for i in range(1, len(centroids)):
        x0, y0 = centroids[i - 1]; x1, y1 = centroids[i]
        if np.isfinite(x0) and np.isfinite(x1):
            u[i] = x1 - x0; v[i] = y1 - y0
    if len(centroids) > 1:
        u[0], v[0] = u[1], v[1]

    feats = np.column_stack([u, v, np.array(A_list), np.array(h_list)])
    valid = np.isfinite(feats).all(axis=1)
    return feats[valid], np.array(dates)[valid]

# 三、时序滑动窗口序列构造

def create_sequences(data: np.ndarray, look_back: int = 12):
    """
    将时序数组转换为 LSTM 输入格式。
      X[i] = data[i : i+look_back]  (历史窗口)
      Y[i] = data[i + look_back]    (预测目标)
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i: i + look_back])
        Y.append(data[i + look_back])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def create_sequences_grouped(data: np.ndarray, look_back: int = 12,
                             seg_ids: np.ndarray = None,
                             return_owner: bool = False):
    if seg_ids is None:
        seg_ids = np.zeros(len(data), dtype=int)
    X, Y, owner = [], [], []
    n = len(data)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seg_ids[j + 1] == seg_ids[i]:
            j += 1
        seg = data[i:j + 1]            # 一个连续段 [i, j]
        for k in range(len(seg) - look_back):
            X.append(seg[k:k + look_back])
            Y.append(seg[k + look_back])
            owner.append(int(seg_ids[i]))
        i = j + 1
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    if return_owner:
        return X, Y, np.array(owner, dtype=int)
    return X, Y


def create_sequences_weighted(data: np.ndarray, weights: np.ndarray,
                              look_back: int = 12):

    X, Y, W = [], [], []
    for i in range(len(data) - look_back):
        X.append(data[i: i + look_back])
        Y.append(data[i + look_back])
        W.append(weights[i + look_back])
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(W, dtype=np.float32))


ALPHA_2021, ALPHA_2025 = 0.018, 0.024    
THETA_2021, THETA_2025 = 25.0, 18.0       
SIC_LOW_THRESHOLD       = 0.70            
SIC_LOW_ALPHA_BOOST     = 1.25           
OMEGA = 7.2921e-5                          


def _alpha_for_year(year: int) -> float:
    """风因子随年份线性插值 (2021->2025)。区间外做线性外推。"""
    frac = (year - 2021) / (2025 - 2021)
    return ALPHA_2021 + frac * (ALPHA_2025 - ALPHA_2021)


def _theta_for_year(year: int) -> float:
    """转向角(度)随年份线性减小。"""
    frac = (year - 2021) / (2025 - 2021)
    return THETA_2021 + frac * (THETA_2025 - THETA_2021)


def _rotate(u: float, v: float, theta_deg: float):
    """将风矢量逆时针旋转 theta 度 (北半球冰相对风右偏取负角)。"""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return c * u - s * v, s * u + c * v


def _synthetic_wind(lon: float, lat: float, t: datetime):
    doy = t.timetuple().tm_yday
    seasonal = 1.0 + 0.4 * np.cos(2 * np.pi * (doy - 30) / 365.0)  # 冬强夏弱
    # 穿极漂流: 大致从西伯利亚一侧吹向 Fram 海峡, 这里给一个经度相关方向场
    base_u = 4.0 * np.cos(np.deg2rad(lon)) * seasonal
    base_v = -3.0 * np.sin(np.deg2rad(lon)) * seasonal - 1.5 * seasonal
    # 确定性扰动 (用坐标+时间做种子, 保证可复现)
    rng = np.random.default_rng(int(abs(lon * 100) + abs(lat * 100) + doy))
    base_u += rng.normal(0, 1.5)
    base_v += rng.normal(0, 1.5)
    return float(base_u), float(base_v)


def _sample_wind_from_nc(wind_nc_dir, lon, lat, t):
    if not wind_nc_dir or not HAS_XARRAY or not os.path.isdir(wind_nc_dir):
        return None
    date_str = t.strftime('%Y%m%d')
    cand = [f for f in os.listdir(wind_nc_dir)
            if f.lower().endswith('.nc') and date_str in f]
    if not cand:
        return None
    ds = _safe_open_dataset(os.path.join(wind_nc_dir, cand[0]))
    if ds is None:
        return None
    try:
        uname = next((n for n in ('u10', 'u', 'U10', 'eastward_wind') if n in ds), None)
        vname = next((n for n in ('v10', 'v', 'V10', 'northward_wind') if n in ds), None)
        if uname is None or vname is None:
            return None
        lat_keys = [k for k in list(ds.coords) + list(ds.data_vars) if 'lat' in k.lower()]
        lon_keys = [k for k in list(ds.coords) + list(ds.data_vars) if 'lon' in k.lower()]
        if not lat_keys or not lon_keys:
            return None
        lats = ds[lat_keys[0]].values
        lons = ds[lon_keys[0]].values
        if lats.ndim == 1:
            i = int(np.argmin(np.abs(lats - lat)))
            j = int(np.argmin(np.abs(lons - lon)))
            uu = np.asarray(ds[uname].values)
            vv = np.asarray(ds[vname].values)
            uu = uu[..., i, j].ravel()[-1]
            vv = vv[..., i, j].ravel()[-1]
        else:
            d = (lats - lat) ** 2 + (lons - lon) ** 2
            i, j = np.unravel_index(np.argmin(d), d.shape)
            uu = np.asarray(ds[uname].values)[..., i, j].ravel()[-1]
            vv = np.asarray(ds[vname].values)[..., i, j].ravel()[-1]
        if np.isfinite(uu) and np.isfinite(vv):
            return float(uu), float(vv)
        return None
    except Exception:
        return None
    finally:
        ds.close()


def _meters_to_degrees(du_m, dv_m, lat):
    """把米位移转成经纬度增量 (近似)。"""
    dlat = dv_m / 111_320.0
    dlon = du_m / (111_320.0 * max(np.cos(np.deg2rad(lat)), 1e-3))
    return dlon, dlat


def simulate_free_drift_pseudolabels(
        start_points, sim_years,
        wind_nc_dir=None,
        integrate_hours=48, substep_hours=6,
        sic_field_fn=None, depth_fn=None,
        samples_per_year=400, seed=42):
    rng = np.random.default_rng(seed)
    n_sub = max(1, int(integrate_hours // substep_hours))
    dt_s  = substep_hours * 3600.0
    out = {}

    for year in sim_years:
        alpha0 = _alpha_for_year(year)
        theta0 = _theta_for_year(year)
        feats, dates = [], []

        for _ in range(samples_per_year):
            lon, lat = start_points[rng.integers(0, len(start_points))]
            lon = float(lon) + float(rng.normal(0, 0.5))
            lat = float(lat) + float(rng.normal(0, 0.3))
            doy = int(rng.integers(1, 365))
            t   = datetime(year, 1, 1) + timedelta(days=doy)

            for _ in range(n_sub):
                # 1) 取风场
                w = _sample_wind_from_nc(wind_nc_dir, lon, lat, t)
                if w is None:
                    w = _synthetic_wind(lon, lat, t)
                u_w, v_w = w

                # 2) SIC -> 调整风因子
                if sic_field_fn is not None:
                    sic = float(sic_field_fn(lon, lat, t))
                else:
                    sic = float(rng.uniform(0.5, 0.95))
                alpha = alpha0 * (SIC_LOW_ALPHA_BOOST if sic < SIC_LOW_THRESHOLD else 1.0)

                # 3) 自由漂移: 冰速 = alpha * R(theta) * 风  (m/s)
                u_ice, v_ice = _rotate(u_w, v_w, -theta0)  # 北半球右偏取负角
                u_ice *= alpha
                v_ice *= alpha

                # 4) 冰厚/水深特征 (供 5 维)
                h = depth_fn(lon, lat) if depth_fn else 1.8 - 0.05 * (year - 2021)  # 逐年变薄
                depth = depth_fn(lon, lat) if depth_fn else 1500.0

                # 记录该子步特征 (u,v 单位 m/s, 与统一量纲一致)
                feats.append([u_ice, v_ice, sic, float(h), float(depth)])
                dates.append(t.strftime('%Y-%m-%d %H:%M'))

                # 5) 欧拉积分前推位置
                du_m, dv_m = u_ice * dt_s, v_ice * dt_s
                dlon, dlat = _meters_to_degrees(du_m, dv_m, lat)
                lon += dlon
                lat = float(np.clip(lat + dlat, -89.5, 89.5))
                t  += timedelta(hours=substep_hours)

        out[year] = (np.array(feats, dtype=float), np.array(dates))
        logging.info(f'[模拟] {year} 年: alpha0={alpha0:.4f}, theta0={theta0:.1f}deg, '
                     f'生成 {len(feats)} 个伪标签时序点')
    return out


def make_start_points_from_history(data, dates):
    pts = [(-150, 75), (-140, 78), (160, 80), (140, 82),
           (100, 81), (60, 83), (0, 84), (-60, 80), (-90, 77)]
    return pts

DRIFT_UNIT_TO_MS = 1000.0 / (2 * 86400.0)

def _compute_true_uv_from_osisaf(ds, feat_dim=4, default_A=0.85, default_h=1.8):
    needed = ('lat', 'lon', 'lat1', 'lon1')
    if not all(n in ds for n in needed):
        return None  
    lat0 = np.asarray(ds['lat'].values, dtype=float)         # 2D 起点
    lon0 = np.asarray(ds['lon'].values, dtype=float)
    lat1 = np.asarray(ds['lat1'].values, dtype=float).squeeze()  # 终点
    lon1 = np.asarray(ds['lon1'].values, dtype=float).squeeze()

    
    mask = (np.isfinite(lat1) & np.isfinite(lon1) &
            (lat1 > -990) & (lon1 > -990))
    if mask.sum() < 5:
        return None

    R = 6371000.0
    dlat = np.deg2rad(lat1 - lat0)
    dlon_deg = lon1 - lon0
    dlon_deg = (dlon_deg + 180.0) % 360.0 - 180.0   
    dlon = np.deg2rad(dlon_deg)
    lat_mean = np.deg2rad((lat0 + lat1) / 2.0)
    east_m  = R * np.cos(lat_mean) * dlon            
    north_m = R * dlat                                

    dt_s = 2 * 86400.0   
    u = east_m / dt_s
    v = north_m / dt_s


    u_ms = float(np.mean(u[mask]))
    v_ms = float(np.mean(v[mask]))

    
    weight = 0.6
    if 'uncert_dX_and_dY' in ds:
        unc = np.asarray(ds['uncert_dX_and_dY'].values, dtype=float).squeeze()
        unc_valid = unc[mask & np.isfinite(unc)]
        if len(unc_valid):
            sigma_ms = abs(float(np.mean(unc_valid)) * DRIFT_UNIT_TO_MS)
            scale = max(abs(u_ms) + abs(v_ms), 1e-3)
            weight = 1.0 / (1.0 + (sigma_ms / scale) ** 2)

    return u_ms, v_ms, float(np.clip(weight, 0.05, 1.0))


def load_drift_weak_labels(drift_nc_dir, sim_years,
                           target_lat=None, target_lon=None,
                           unit_to_ms=DRIFT_UNIT_TO_MS,
                           feat_dim=4, default_A=0.85, default_h=1.8):
   
    if not HAS_XARRAY or not drift_nc_dir or not os.path.isdir(drift_nc_dir):
        logging.warning('[弱标签] xarray 不可用或目录无效, 跳过弱标签加载。')
        return {}

    files = sorted(f for f in os.listdir(drift_nc_dir) if f.lower().endswith('.nc'))
    if not files:
        logging.warning(f'[弱标签] {drift_nc_dir} 中无 .nc 文件。')
        return {}

    out = {y: ([], [], []) for y in sim_years}

    for fn in files:
        m8 = re.search(r'(\d{8})', fn)
        m4 = re.search(r'(\d{4})', fn)
        if m8:
            year = int(m8.group(1)[:4]); date_str = m8.group(1)
        elif m4:
            year = int(m4.group(1)); date_str = m4.group(1)
        else:
            continue
        if year not in out:
            continue

        ds = _safe_open_dataset(os.path.join(drift_nc_dir, fn))
        if ds is None:
            continue
        try:
            # --- 优先: OSI SAF 真东北速度法 ---
            res = _compute_true_uv_from_osisaf(ds, feat_dim=feat_dim,
                                               default_A=default_A, default_h=default_h)
            if res is not None:
                u_ms, v_ms, weight = res
            else:
                # --- 回退: 直接读 dX/dY 等 (其他产品格式) ---
                u_cands = ('dX', 'dx', 'u', 'eastward_sea_ice_velocity')
                v_cands = ('dY', 'dy', 'v', 'northward_sea_ice_velocity')
                uname = next((n for n in u_cands if n in ds), None)
                vname = next((n for n in v_cands if n in ds), None)
                if uname is None or vname is None:
                    continue
                u_ms = float(np.nanmean(ds[uname].values)) * unit_to_ms
                v_ms = float(np.nanmean(ds[vname].values)) * unit_to_ms
                weight = 0.6

            if not (np.isfinite(u_ms) and np.isfinite(v_ms)):
                continue

            
            row = [u_ms, v_ms]
            extras = [default_A, default_h, 1500.0]  
            row += extras[:max(0, feat_dim - 2)]
            row = row[:feat_dim]

            feats, dates, weights = out[year]
            feats.append(row)
            dates.append(date_str)
            weights.append(weight)
        except Exception as e:
            logging.debug(f'[弱标签] 读取 {fn} 失败: {e}')
        finally:
            ds.close()

    result = {}
    for y in sim_years:
        f, d, w = out[y]
        if len(f) >= 2:
            arr = np.array(f, dtype=float)
            result[y] = (arr, np.array(d), np.array(w, dtype=float))
            logging.info(f'[弱标签] {y} 年: {len(f)} 个标签, 平均权重={np.mean(w):.3f}, '
                         f'u均值={arr[:,0].mean():.4f} m/s, v均值={arr[:,1].mean():.4f} m/s')
        else:
            logging.warning(f'[弱标签] {y} 年标签不足 ({len(f)}), 跳过。')
    return result


def build_pinn_lstm(look_back: int, feat_dim: int) -> tf.keras.Model:
    inputs  = tf.keras.Input(shape=(look_back, feat_dim), name='seq_input')
    h       = LSTM(64, return_sequences=True, name='lstm_1')(inputs)
    h       = Dropout(0.3, name='drop_1')(h)
    h       = LSTM(32, return_sequences=False, name='lstm_2')(h)
    h       = Dropout(0.3, name='drop_2')(h)
    outputs = Dense(feat_dim, name='output')(h)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='PI_LSTM')


def make_train_step(model, optimizer, data_scale_tf, data_mean_tf,
                    t_std_tf, pinn_weight: float):
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred_norm = model(x_batch, training=True)
            data_loss   = tf.reduce_mean(tf.square(y_pred_norm - y_batch))

            y_pred_phys = (y_pred_norm - data_mean_tf) / data_scale_tf          
            x_last_phys = (x_batch[:, -1, :] - data_mean_tf) / data_scale_tf    

            u_pred = y_pred_phys[:, 0]
            v_pred = y_pred_phys[:, 1]
            A_pred = y_pred_phys[:, 2]
            h_pred = y_pred_phys[:, 3]
            

            du_dt = (u_pred - x_last_phys[:, 0]) / t_std_tf
            dv_dt = (v_pred - x_last_phys[:, 1]) / t_std_tf
            dA_dt = (A_pred - x_last_phys[:, 2]) / t_std_tf
            dh_dt = (h_pred - x_last_phys[:, 3]) / t_std_tf

           
            air_forcing   = 0.05 * A_pred
            ocean_forcing = 0.02 * A_pred
            drag_coef     = 0.01
            coriolis_coef = 0.005

            res_u = du_dt - (air_forcing   - drag_coef * u_pred - coriolis_coef * v_pred)
            res_v = dv_dt - (ocean_forcing - drag_coef * v_pred + coriolis_coef * u_pred)
            res_A = dA_dt - 0.001 * (1.0 - A_pred)
            res_h = dh_dt - 0.002 * (1.0 - h_pred)

            phys_loss = (
                tf.reduce_mean(tf.square(res_u)) +
                tf.reduce_mean(tf.square(res_v)) +
                0.5 * tf.reduce_mean(tf.square(res_A)) +
                0.5 * tf.reduce_mean(tf.square(res_h))
            )

           
            total_loss = (1.0 - pinn_weight) * data_loss + pinn_weight * phys_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, data_loss, phys_loss

    return train_step


def compute_fisher(model, X, y, batch_size=16):
    """
    在旧任务(收敛后)估计 Fisher 信息对角线, 用于 EWC。
    Fisher_i ≈ E[(d log p / d theta_i)^2]，这里用 MSE 梯度平方近似。
    返回 list[tf.Tensor]，与 model.trainable_variables 对齐。
    """
    fisher = [tf.zeros_like(v) for v in model.trainable_variables]
    n = 0
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    for xb, yb in ds:
        with tf.GradientTape() as tape:
            pred = model(xb, training=False)
            loss = tf.reduce_mean(tf.square(pred - yb))
        grads = tape.gradient(loss, model.trainable_variables)
        for i, g in enumerate(grads):
            if g is not None:
                fisher[i] += tf.square(g) * tf.cast(tf.shape(xb)[0], tf.float32)
        n += int(xb.shape[0])
    if n > 0:
        fisher = [f / float(n) for f in fisher]
    return fisher


def make_cl_train_step(model, optimizer, data_scale_tf, data_mean_tf, t_std_tf,
                       pinn_weight, ewc_lambda, fisher, star_vars):
    base_step = None  # 占位, 逻辑内联以共享 GradientTape

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred_norm = model(x_batch, training=True)
            data_loss   = tf.reduce_mean(tf.square(y_pred_norm - y_batch))

            y_pred_phys = (y_pred_norm - data_mean_tf) / data_scale_tf          # 修复: MinMax逆变换
            x_last_phys = (x_batch[:, -1, :] - data_mean_tf) / data_scale_tf    # 修复: MinMax逆变换

            u_pred = y_pred_phys[:, 0]; v_pred = y_pred_phys[:, 1]
            A_pred = y_pred_phys[:, 2]; h_pred = y_pred_phys[:, 3]

            du_dt = (u_pred - x_last_phys[:, 0]) / t_std_tf
            dv_dt = (v_pred - x_last_phys[:, 1]) / t_std_tf
            dA_dt = (A_pred - x_last_phys[:, 2]) / t_std_tf
            dh_dt = (h_pred - x_last_phys[:, 3]) / t_std_tf

            air_forcing = 0.05 * A_pred; ocean_forcing = 0.02 * A_pred
            drag_coef = 0.01; coriolis_coef = 0.005
            res_u = du_dt - (air_forcing   - drag_coef * u_pred - coriolis_coef * v_pred)
            res_v = dv_dt - (ocean_forcing - drag_coef * v_pred + coriolis_coef * u_pred)
            res_A = dA_dt - 0.001 * (1.0 - A_pred)
            res_h = dh_dt - 0.002 * (1.0 - h_pred)
            phys_loss = (tf.reduce_mean(tf.square(res_u)) +
                         tf.reduce_mean(tf.square(res_v)) +
                         0.5 * tf.reduce_mean(tf.square(res_A)) +
                         0.5 * tf.reduce_mean(tf.square(res_h)))

            total_loss = (1.0 - pinn_weight) * data_loss + pinn_weight * phys_loss

           
            ewc_loss = tf.constant(0.0, dtype=tf.float32)
            if ewc_lambda > 0 and fisher is not None and star_vars is not None:
                for f, star, var in zip(fisher, star_vars, model.trainable_variables):
                    ewc_loss += tf.reduce_sum(f * tf.square(var - star))
                total_loss += 0.5 * ewc_lambda * ewc_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, data_loss, phys_loss, ewc_loss

    return train_step


def continual_learning_loop(
        model, optimizer, old_scaled, sim_by_year,
        scaler, look_back, feat_dim,
        data_scale_tf, data_mean_tf, t_std_tf,
        pinn_weight, replay_ratio=0.15, ewc_lambda=0.0,
        epochs_per_year=20, batch_size=16, seed=42):

    rng = np.random.default_rng(seed)

    # --- 记忆集 (旧任务序列) ---
    X_old, y_old = create_sequences(old_scaled, look_back=look_back)
    n_mem = max(1, int(replay_ratio * len(X_old)))
    mem_idx = rng.choice(len(X_old), size=n_mem, replace=False)
    X_mem, y_mem = X_old[mem_idx], y_old[mem_idx]
    logging.info(f'[CL] 记忆集大小: {len(X_mem)} / 旧数据 {len(X_old)} '
                 f'(replay_ratio={replay_ratio})')

    fisher = star_vars = None  # EWC 状态

    for k, year in enumerate(sorted(sim_by_year.keys())):
        sim_feats, _ = sim_by_year[year]
        if len(sim_feats) < look_back + 2:
            logging.warning(f'[CL] {year} 年模拟样本不足, 跳过')
            continue

        sim_scaled = scaler.transform(sim_feats)
        X_new, y_new = create_sequences(sim_scaled, look_back=look_back)

        reps = max(1, len(X_new) // max(1, len(X_mem)))
        X_mem_rep = np.repeat(X_mem, reps, axis=0)
        y_mem_rep = np.repeat(y_mem, reps, axis=0)

        X_mix = np.concatenate([X_new, X_mem_rep], axis=0)
        y_mix = np.concatenate([y_new, y_mem_rep], axis=0)
        logging.info(f'[CL] 年份 {year}: 新数据 {len(X_new)} + 记忆(上采样) '
                     f'{len(X_mem_rep)} = 混合 {len(X_mix)}')

        cl_step = make_cl_train_step(
            model, optimizer, data_scale_tf, data_mean_tf, t_std_tf,
            pinn_weight, ewc_lambda, fisher, star_vars
        )

        ds = (tf.data.Dataset.from_tensor_slices((X_mix, y_mix))
              .shuffle(len(X_mix), seed=seed).batch(batch_size))

        for epoch in range(epochs_per_year):
            e_tot = e_dat = e_phy = e_ewc = 0.0; n = 0
            for xb, yb in ds:
                t_l, d_l, p_l, w_l = cl_step(xb, yb)
                e_tot += float(t_l); e_dat += float(d_l)
                e_phy += float(p_l); e_ewc += float(w_l); n += 1
            if (epoch + 1) % max(1, epochs_per_year // 2) == 0 or epoch == 0:
                logging.info(f'  [CL {year}] epoch {epoch+1}/{epochs_per_year} | '
                             f'Total {e_tot/n:.6f} | Data {e_dat/n:.6f} | '
                             f'Phys {e_phy/n:.6f} | EWC {e_ewc/max(n,1):.6f}')

        
        if ewc_lambda > 0 and fisher is None:
            logging.info(f'[CL] 在旧数据上计算 Fisher 信息 (EWC, lambda={ewc_lambda}) ...')
            fisher = compute_fisher(model, X_old, y_old, batch_size=batch_size)
            star_vars = [tf.identity(v) for v in model.trainable_variables]

    return model


def make_weighted_cl_step(model, optimizer, data_scale_tf, data_mean_tf, t_std_tf,
                          pinn_weight, ewc_lambda, fisher, star_vars):
    @tf.function
    def train_step(x_batch, y_batch, w_batch):
        with tf.GradientTape() as tape:
            y_pred_norm = model(x_batch, training=True)
            # 加权 MSE: w_batch 形状 (n, feat_dim) — 逐样本逐维权重。
            # 弱标签样本仅 u/v 维有监督 (A/h 为占位, 掩码=0); 记忆样本全维=1。
            sq = tf.square(y_pred_norm - y_batch)
            w_sum = tf.reduce_sum(w_batch, axis=-1) + 1e-8
            per_sample = tf.reduce_sum(sq * w_batch, axis=-1) / w_sum
            data_loss = tf.reduce_mean(per_sample)

            y_pred_phys = (y_pred_norm - data_mean_tf) / data_scale_tf          # 修复: MinMax逆变换
            x_last_phys = (x_batch[:, -1, :] - data_mean_tf) / data_scale_tf    # 修复: MinMax逆变换
            u_pred = y_pred_phys[:, 0]; v_pred = y_pred_phys[:, 1]
            A_pred = y_pred_phys[:, 2]; h_pred = y_pred_phys[:, 3]
            du_dt = (u_pred - x_last_phys[:, 0]) / t_std_tf
            dv_dt = (v_pred - x_last_phys[:, 1]) / t_std_tf
            dA_dt = (A_pred - x_last_phys[:, 2]) / t_std_tf
            dh_dt = (h_pred - x_last_phys[:, 3]) / t_std_tf
            air_forcing = 0.05 * A_pred; ocean_forcing = 0.02 * A_pred
            drag_coef = 0.01; coriolis_coef = 0.005
            res_u = du_dt - (air_forcing   - drag_coef * u_pred - coriolis_coef * v_pred)
            res_v = dv_dt - (ocean_forcing - drag_coef * v_pred + coriolis_coef * u_pred)
            res_A = dA_dt - 0.001 * (1.0 - A_pred)
            res_h = dh_dt - 0.002 * (1.0 - h_pred)
            phys_loss = (tf.reduce_mean(tf.square(res_u)) +
                         tf.reduce_mean(tf.square(res_v)) +
                         0.5 * tf.reduce_mean(tf.square(res_A)) +
                         0.5 * tf.reduce_mean(tf.square(res_h)))

            total_loss = (1.0 - pinn_weight) * data_loss + pinn_weight * phys_loss

            ewc_loss = tf.constant(0.0, dtype=tf.float32)
            if ewc_lambda > 0 and fisher is not None and star_vars is not None:
                for f, star, var in zip(fisher, star_vars, model.trainable_variables):
                    ewc_loss += tf.reduce_sum(f * tf.square(var - star))
                total_loss += 0.5 * ewc_lambda * ewc_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, data_loss, phys_loss, ewc_loss

    return train_step


def weak_label_finetune_loop(
        model, optimizer, old_scaled, weak_by_year,
        scaler, look_back, feat_dim,
        data_scale_tf, data_mean_tf, t_std_tf,
        pinn_weight, replay_ratio=0.15, ewc_lambda=0.0,
        epochs_per_year=20, batch_size=16, seed=42):

    rng = np.random.default_rng(seed)
    X_old, y_old = create_sequences(old_scaled, look_back=look_back)
    n_mem = max(1, int(replay_ratio * len(X_old)))
    mem_idx = rng.choice(len(X_old), size=n_mem, replace=False)
    X_mem, y_mem = X_old[mem_idx], y_old[mem_idx]
    logging.info(f'[弱标签CL] 记忆集: {len(X_mem)} / 旧数据 {len(X_old)}')

   
    old_phys_mean = scaler.inverse_transform(old_scaled).mean(axis=0)

    fisher = [tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
    star_vars = [tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
    ewc_ready = False  

    for year in sorted(weak_by_year.keys()):
        feats, _, weights = weak_by_year[year]
        if len(feats) < 2:
            logging.warning(f'[弱标签CL] {year} 年标签不足 (<2), 跳过')
            continue

        feats = feats.copy()
        if feats.shape[1] > 2:
            feats[:, 2:] = old_phys_mean[2:feats.shape[1]]

    
        if len(feats) < look_back + 2:
            n_target = max(look_back + 10, len(feats) * 6)
            xi = np.linspace(0, len(feats) - 1, n_target)
            x0 = np.arange(len(feats))
            feats = np.column_stack([np.interp(xi, x0, feats[:, k])
                                     for k in range(feats.shape[1])])
            weights = np.interp(xi, x0, weights)
            logging.info(f'[弱标签CL] {year}: 月度标签插值加密 -> {len(feats)} 点')

        scaled_y = scaler.transform(feats)
        Xw, yw, ww_scalar = create_sequences_weighted(scaled_y, weights, look_back=look_back)

        
        dim_mask_weak = np.zeros(feat_dim, dtype=np.float32)
        dim_mask_weak[:2] = 1.0
        ww = ww_scalar[:, None] * dim_mask_weak[None, :]        


        reps = max(1, len(Xw) // max(1, len(X_mem)))
        X_mem_r = np.repeat(X_mem, reps, axis=0)
        y_mem_r = np.repeat(y_mem, reps, axis=0)
        w_mem_r = np.ones((len(X_mem_r), feat_dim), dtype=np.float32)

        X_mix = np.concatenate([Xw, X_mem_r], axis=0)
        y_mix = np.concatenate([yw, y_mem_r], axis=0)
        w_mix = np.concatenate([ww, w_mem_r], axis=0).astype(np.float32)
        logging.info(f'[弱标签CL] {year}: 弱标签 {len(Xw)}(均权{ww_scalar.mean():.2f}, '
                     f'仅监督u/v) + 记忆 {len(X_mem_r)} = {len(X_mix)}')

        step = make_weighted_cl_step(
            model, optimizer, data_scale_tf, data_mean_tf, t_std_tf,
            pinn_weight, ewc_lambda, fisher, star_vars)

        ds = (tf.data.Dataset.from_tensor_slices((X_mix, y_mix, w_mix))
              .shuffle(len(X_mix), seed=seed).batch(batch_size))

        for epoch in range(epochs_per_year):
            e_tot = e_dat = e_phy = e_ewc = 0.0; n = 0
            for xb, yb, wb in ds:
                t_l, d_l, p_l, w_l = step(xb, yb, wb)
                e_tot += float(t_l); e_dat += float(d_l)
                e_phy += float(p_l); e_ewc += float(w_l); n += 1
            if (epoch + 1) % max(1, epochs_per_year // 2) == 0 or epoch == 0:
                logging.info(f'  [弱标签 {year}] epoch {epoch+1}/{epochs_per_year} | '
                             f'Total {e_tot/n:.6f} | Data {e_dat/n:.6f} | '
                             f'Phys {e_phy/n:.6f} | EWC {e_ewc/max(n,1):.6f}')

        if ewc_lambda > 0 and not ewc_ready:
            logging.info(f'[弱标签CL] 计算 Fisher (EWC, lambda={ewc_lambda})...')
            new_fisher = compute_fisher(model, X_old, y_old, batch_size=batch_size)
            for fv, nf in zip(fisher, new_fisher):
                fv.assign(nf)
            for sv, v in zip(star_vars, model.trainable_variables):
                sv.assign(v)
            ewc_ready = True
            fsum = float(sum(float(tf.reduce_sum(f)) for f in fisher))
            logging.info(f'[诊断] Fisher总和={fsum:.6e}')
    return model


def _phys_bounds(feat_dim):
    """递归预测时的物理边界 (反归一化空间)。A∈[0,1], h≥0; u/v 不限。"""
    pmin = np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0][:feat_dim], dtype=np.float32)
    pmax = np.array([ np.inf,  np.inf, 1.0, np.inf, np.inf][:feat_dim], dtype=np.float32)
    return pmin, pmax


def _clip_to_scaled(p_norm, scaler, pmin, pmax, static_depth=None):
    """归一化空间 -> 反归一化 clip 到物理边界 -> 再归一化; 静态深度强制还原。"""
    p_phys = scaler.inverse_transform(p_norm.reshape(1, -1))[0]
    p_phys = np.clip(p_phys, pmin, pmax)
    p_norm = scaler.transform(p_phys.reshape(1, -1))[0].astype(np.float32)
    if static_depth is not None and len(p_norm) >= 5:
        p_norm[4] = static_depth
    return p_norm, p_phys


def predict_mc_dropout(model, seq_init, scaler, look_back, feat_dim,
                       predict_steps, mc_samples=50):

    pmin, pmax = _phys_bounds(feat_dim)
    static_depth = seq_init[-1, 4] if feat_dim >= 5 else None

    all_runs = np.zeros((mc_samples, predict_steps, feat_dim), dtype=float)
    for s in range(mc_samples):
        seq = seq_init.copy()
        for t in range(predict_steps):
            inp = seq.reshape(1, look_back, feat_dim)
            # training=True -> Dropout 生效, 引入随机性
            p = model(inp, training=True).numpy()[0]
            p_norm, p_phys = _clip_to_scaled(p, scaler, pmin, pmax, static_depth)
            all_runs[s, t] = p_phys
            seq = np.vstack([seq[1:], p_norm])
    return all_runs.mean(axis=0), all_runs.std(axis=0)


def predict_ensemble(models, seq_init, scaler, look_back, feat_dim, predict_steps):

    pmin, pmax = _phys_bounds(feat_dim)
    static_depth = seq_init[-1, 4] if feat_dim >= 5 else None

    M = len(models)
    all_runs = np.zeros((M, predict_steps, feat_dim), dtype=float)
    for m, mdl in enumerate(models):
        seq = seq_init.copy()
        for t in range(predict_steps):
            inp = seq.reshape(1, look_back, feat_dim)
            p = mdl(inp, training=False).numpy()[0]
            p_norm, p_phys = _clip_to_scaled(p, scaler, pmin, pmax, static_depth)
            all_runs[m, t] = p_phys
            seq = np.vstack([seq[1:], p_norm])
    return all_runs.mean(axis=0), all_runs.std(axis=0)


def fit_train_distribution(scaled_train):

    mu = np.mean(scaled_train, axis=0)
    cov = np.cov(scaled_train, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-4  # 数值稳定
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    return mu, inv_cov


def mahalanobis_ood(x_scaled, mu, inv_cov):
    d = x_scaled - mu
    return float(np.sqrt(max(d @ inv_cov @ d, 0.0)))


def assess_ood(seq_init_scaled, mu, inv_cov, threshold=3.0):

    flags = []
    for i, x in enumerate(seq_init_scaled):
        dist = mahalanobis_ood(x, mu, inv_cov)
        flags.append((i, dist, dist > threshold))
    return flags

def get_user_input(default_lon=None, default_lat=None,
                   default_steps=None, default_days=None) -> dict:
    print("\n" + "=" * 50)
    print("  海冰漂移预测系统 — 参数输入")
    print("=" * 50)

    def _read_float(prompt, lo, hi, default):
        while True:
            try:
                s   = input(prompt).strip()
                val = default if (s == "" and default is not None) else float(s)
                if lo <= val <= hi:
                    return val
                print(f"  值应在 [{lo}, {hi}] 范围内")
            except ValueError:
                print("  请输入有效数字")

    def _read_int(prompt, default):
        while True:
            try:
                s   = input(prompt).strip()
                val = default if (s == "" and default is not None) else int(s)
                if val > 0:
                    return val
                print("  须为正整数")
            except ValueError:
                print("  请输入有效整数")

    lon   = _read_float(f"经度 (默认 {default_lon}): " if default_lon else "经度 (-180~180): ",
                        -180, 360, default_lon)
    lat   = _read_float(f"纬度 (默认 {default_lat}): " if default_lat else "纬度 (-90~90): ",
                        -90, 90, default_lat)
    steps = _read_int(f"预测天数 (默认 {default_steps or 7}): ", default_steps or 7)
    days  = _read_int(f"历史天数 (默认 {default_days or 365}): ", default_days or 365)

    return {'longitude': lon, 'latitude': lat,
            'predict_steps': steps, 'history_days': days}



def print_prediction_result(predictions: np.ndarray, dates_pred: list,
                             lon: float, lat: float):
    """格式化打印预测结果表格"""
    feat_dim = predictions.shape[1]
    headers  = ['u(东西速度)', 'v(南北速度)', 'A(密集度)', 'h(冰厚度)', 'depth(水深)']

    print("\n" + "=" * 72)
    print(f"  预测结果 — 经度: {lon:.4f}, 纬度: {lat:.4f}")
    print("=" * 72)
    print(f"{'日期':<22}" + "".join(f"{h:<12}" for h in headers[:feat_dim]))
    print("-" * 72)
    for pred, dt in zip(predictions, dates_pred):
        print(f"{dt:<22}" + "".join(f"{v:<12.5f}" for v in pred[:feat_dim]))
    print("-" * 72)
    print("\n说明: u 正=东向  v 正=北向  A∈[0,1](1=全覆盖)  h 单位=米")


def export_json(data: np.ndarray, preds_inv: np.ndarray,
                dates_pred: list, log_dir: str, ts: str,
                predict_steps: int, pred_std: np.ndarray = None):

    feat_keys = ['u', 'v', 'A', 'h', 'depth'][:data.shape[1]]

    predicted = []
    for t, pt in enumerate(preds_inv):
        rec = {k: float(pt[i]) for i, k in enumerate(feat_keys)}
        if pred_std is not None:
            for i, k in enumerate(feat_keys):
                rec[f'{k}_std']      = float(pred_std[t, i])
                rec[f'{k}_ci_lower'] = float(pt[i] - 1.96 * pred_std[t, i])
                rec[f'{k}_ci_upper'] = float(pt[i] + 1.96 * pred_std[t, i])
        predicted.append(rec)

    json_output = {
        "history":   [{k: float(pt[i]) for i, k in enumerate(feat_keys)}
                      for pt in data[-100:]],
        "predicted": predicted,
        "meta": {
            "generated_at":      datetime.now(timezone.utc).isoformat() + "Z",
            "predict_steps":     predict_steps,
            "prediction_dates":  dates_pred,
            "has_uncertainty":   pred_std is not None,
        }
    }

    for fname in [f'drift_data_{ts}.json', 'drift_data_latest.json']:
        fpath = os.path.join(log_dir, fname)
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            logging.info(f'JSON 导出: {fpath}')
        except Exception as e:
            logging.warning(f'JSON 导出失败 {fpath}: {e}')


def interactive_viewer(nc_dir: str = '.', varname: str = None,
                       cmap: str = 'viridis', interval: int = 500):
    if not HAS_XARRAY:
        raise ImportError('xarray 未安装，无法使用查看器。pip install xarray netcdf4')

    from matplotlib.widgets import Slider, Button, TextBox

    files = sorted(f for f in os.listdir(nc_dir) if f.lower().endswith('.nc'))
    if not files:
        raise FileNotFoundError(f'在 {nc_dir} 中未找到 .nc 文件')

    filepaths = [os.path.join(nc_dir, f) for f in files]
    dates     = [re.search(r'(\d{8})', f).group(1)
                 if re.search(r'(\d{8})', f) else f for f in files]
    cache     = {}

    def _choose_var(ds, preferred=None):
        if preferred and preferred in ds.data_vars:
            return preferred
        for name, var in ds.data_vars.items():
            if len(var.dims) == 2:
                return name
        return list(ds.data_vars.keys())[0]

    def _load(idx):
        if idx in cache:
            return cache[idx]
        with xr.open_dataset(filepaths[idx]) as ds:
            sv = varname or _choose_var(ds)
            da = ds[sv].squeeze()
            while da.ndim > 2:
                da = da.isel({da.dims[0]: 0})
            arr = da.values.astype(float)
            arr[~np.isfinite(arr)] = np.nan
            cache[idx] = (arr, sv)
        return cache[idx]

    arr0, var0 = _load(0)
    fig, ax    = plt.subplots(figsize=(9, 7))
    im         = ax.imshow(arr0, origin='upper', cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'{dates[0]}   [{var0}]')

    slider   = Slider(plt.axes([0.12, 0.02, 0.58, 0.03], facecolor='#e8e8e8'),
                      'Frame', 0, len(files) - 1, valinit=0, valstep=1)
    btn_prev = Button(plt.axes([0.72, 0.02, 0.06, 0.03]), '◀')
    btn_next = Button(plt.axes([0.79, 0.02, 0.06, 0.03]), '▶')
    btn_play = Button(plt.axes([0.86, 0.02, 0.08, 0.03]), '▶ Play')
    textbox  = TextBox(plt.axes([0.12, 0.06, 0.24, 0.03]), 'Jump', initial='')

    playing = {'state': False}
    timer   = fig.canvas.new_timer(interval=interval)

    def _update(idx):
        arr, var = _load(int(idx))
        im.set_data(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            im.set_clim(finite.min(), finite.max())
        ax.set_title(f'{dates[int(idx)]}   [{var}]')
        fig.canvas.draw_idle()

    def _toggle(e):
        playing['state'] = not playing['state']
        btn_play.label.set_text('⏸ Pause' if playing['state'] else '▶ Play')
        timer.start() if playing['state'] else timer.stop()

    slider.on_changed(_update)
    btn_prev.on_clicked(lambda e: slider.set_val(max(0, int(slider.val) - 1)))
    btn_next.on_clicked(lambda e: slider.set_val(min(len(files) - 1, int(slider.val) + 1)))
    btn_play.on_clicked(_toggle)
    timer.add_callback(
        lambda: slider.set_val((int(slider.val) + 1) % len(files))
        if playing['state'] else None
    )
    textbox.on_submit(
        lambda t: slider.set_val(dates.index(t.strip()))
        if t.strip() in dates
        else print(f'日期 "{t.strip()}" 未找到')
    )
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='海冰漂移 PI-LSTM 预测系统 — 多源数据融合版',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--excel_path', default='E:/dataset/excel_data/drift.xlsx',
                        help='Excel 轨迹坐标表 (.xlsx)')
    parser.add_argument('--tif_dir',    default='E:/dataset/tif_data',
                        help='TIF 速度/密集度根目录 (子目录: u/ v/ A/)')
    parser.add_argument('--nc_dir',     default='E:/dataset/nc_monthly_data',
                        help='NC 月厚度数据目录 / 备用 NC 目录')
    parser.add_argument('--h5_path',    default='E:/dataset/static_grid/depth.h5',
                        help='静态水深 H5 文件')
    # 备用单 NC 模式
    parser.add_argument('--nc_only',    action='store_true',
                        help='跳过 Excel/TIF/H5，直接从 --nc_dir 读取 NC 文件')
    # dataForIce 真实目录模式 (推荐, 适配用户结构)
    parser.add_argument('--dataforice', action='store_true',
                        help='使用 dataForIce 专用加载器 (浮标xlsx + u/v tif + 月厚度nc -> [u,v,A,h])')
    parser.add_argument('--base_dir',   default='E:/dataForIce',
                        help='dataForIce 根目录 (含 excel_data, tif_data)')
    parser.add_argument('--nc_pattern', default='SIT_25km_monthly_{ym}.nc',
                        help='月厚度 nc 文件名模板, {ym} 替换为 YYYYMM')
    parser.add_argument('--days',       type=int, default=730,
                        help='NC 模式下使用的最近文件数量')
    parser.add_argument('--lon',        type=float, default=None,
                        help='NC 模式点采样经度')
    parser.add_argument('--lat',        type=float, default=None,
                        help='NC 模式点采样纬度')
    # 模型超参数
    parser.add_argument('--look_back',     type=int,   default=12,
                        help='LSTM 历史回溯窗口长度')
    parser.add_argument('--epochs',        type=int,   default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size',    type=int,   default=16,
                        help='批大小')
    parser.add_argument('--lr',            type=float, default=1e-3,
                        help='Adam 学习率')
    parser.add_argument('--predict_steps', type=int,   default=7,
                        help='递归预测未来步数')
    parser.add_argument('--pinn_weight',   type=float, default=0.5,
                        help='物理损失权重 (0=纯数据驱动, 1=纯物理约束)')
    # 运行控制
    parser.add_argument('--log_dir',     default='logs',
                        help='日志与输出目录')
    parser.add_argument('--no_plot',     action='store_true',
                        help='不弹出图窗，将图表保存至 log_dir')
    parser.add_argument('--interactive', action='store_true',
                        help='启用交互式参数输入模式')
    # ---- v2 持续学习 / 物理模拟 参数 ----
    parser.add_argument('--continual', action='store_true',
                        help='启用持续学习: 先用旧数据训练, 再用模拟伪标签逐年 Replay/EWC 更新')
    parser.add_argument('--wind_nc_dir', default=None,
                        help='真实风场(ERA5/CMEMS) NC 目录; 缺省则用合成风场')
    parser.add_argument('--sim_years', type=int, nargs='+',
                        default=[2021, 2022, 2023, 2024, 2025],
                        help='模拟伪标签的年份列表')
    parser.add_argument('--samples_per_year', type=int, default=400,
                        help='每年生成的模拟轨迹条数')
    parser.add_argument('--integrate_hours', type=int, default=48,
                        help='单条轨迹前推总时长 (24~72)')
    parser.add_argument('--substep_hours', type=int, default=6,
                        help='欧拉积分子步长 (小时)')
    parser.add_argument('--replay_ratio', type=float, default=0.15,
                        help='记忆重放: 从旧数据抽取的样本比例 (0.1~0.2)')
    parser.add_argument('--ewc_lambda', type=float, default=0.0,
                        help='EWC 正则强度 (0=仅Replay, >0 启用弹性权重巩固)')
    parser.add_argument('--epochs_per_year', type=int, default=20,
                        help='持续学习中每个模拟年份的训练轮数')
    # ---- v3: 弱标签 / 不确定性 / OOD 参数 ----
    parser.add_argument('--weak_label', action='store_true',
                        help='启用弱标签微调: 用低分辨率漂移产品做监督信号 (优于纯模拟)')
    parser.add_argument('--drift_nc_dir', default=None,
                        help='低分辨率漂移产品 NC 目录 (OSI SAF / NSIDC)')
    parser.add_argument('--drift_unit_to_ms', type=float, default=DRIFT_UNIT_TO_MS,
                        help='漂移产品原始值->m/s 换算系数 (km/2day 默认; 若已是m/s设1.0)')
    parser.add_argument('--uncertainty', choices=['none', 'mc_dropout', 'ensemble'],
                        default='none', help='预测不确定性估计方法')
    parser.add_argument('--mc_samples', type=int, default=50,
                        help='MC Dropout 前向采样次数')
    parser.add_argument('--ensemble_size', type=int, default=5,
                        help='Deep Ensemble 模型数量')
    parser.add_argument('--ood_threshold', type=float, default=3.0,
                        help='OOD 马氏距离预警阈值')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts      = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile = os.path.join(args.log_dir, f'seaice_run_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(logfile, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info('=' * 60)
    logging.info('海冰漂移 PI-LSTM 预测系统 启动')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f'GPU 已启用: {[g.name for g in gpus]}')
    else:
        logging.info('当前为 CPU 训练。提示: 原生 Windows 上 TF>=2.11 不支持 GPU '
                     '(TensorFlow 官方限制), 如需 GPU 请在 WSL2 中运行, '
                     '或 pip install tensorflow-directml-plugin (DirectML 方案)。')
    logging.info(f'日志文件: {logfile}')
    logging.info('=' * 60)

    target_lat    = target_lon = None
    predict_steps = args.predict_steps

    if args.interactive:
        try:
            ui = get_user_input(
                default_lon=args.lon, default_lat=args.lat,
                default_steps=args.predict_steps, default_days=args.days
            )
            target_lon    = ui['longitude']
            target_lat    = ui['latitude']
            predict_steps = ui['predict_steps']
            args.days     = ui['history_days']
        except KeyboardInterrupt:
            logging.info('交互式输入中断，使用命令行参数继续运行。')
            if args.lon and args.lat:
                target_lon, target_lat = args.lon, args.lat
    elif args.lon is not None and args.lat is not None:
        target_lon, target_lat = args.lon, args.lat

    logging.info('开始加载数据...')
    seg_ids = None
    try:
        if args.dataforice:
            logging.info(f'[模式] dataForIce 专用加载器: {args.base_dir}')
            # 若用户未显式指定 nc_dir (仍为默认值), 回退到 base_dir/nc_data
            _nc = args.nc_dir
            if _nc == 'E:/dataset/nc_monthly_data':
                _nc = os.path.join(args.base_dir, 'nc_data')
            data, dates, seg_ids = build_dataset_from_dataForIce(
                base_dir=args.base_dir,
                nc_dir=_nc,
                nc_pattern=args.nc_pattern,
            )
        elif args.nc_only:
            logging.info(f'[模式] 单 NC 目录: {args.nc_dir}')
            data, dates = build_dataset_from_nc_dir(
                args.nc_dir, days=args.days,
                target_lat=target_lat, target_lon=target_lon
            )
        else:
            logging.info('[模式] 多源融合: Excel + TIF + NC + H5')
            data, dates = build_dataset_from_mixed_sources(
                args.excel_path, args.tif_dir, args.nc_dir, args.h5_path
            )
    except Exception as e:
        logging.error(f'数据加载失败: {e}')
        raise

    if len(data) < args.look_back + 2:
        raise RuntimeError(
            f'有效时间步 ({len(data)}) 不足，需至少 {args.look_back + 2} 步。'
        )
    logging.info(f'数据形状: {data.shape}，时间范围: {dates[0]} → {dates[-1]}')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y, owner = create_sequences_grouped(
        scaled, look_back=args.look_back, seg_ids=seg_ids, return_owner=True)
    if len(X) == 0:
        raise RuntimeError(
            f'段内滑窗后无可用样本: 多半是切段后每段都短于 look_back+1={args.look_back+1}。'
            f'可减小 --look_back 或放宽切段阈值。')


    uniq_segs = np.unique(owner)
    if seg_ids is not None and len(uniq_segs) > 1:
        n_seg_train = max(1, int(0.8 * len(uniq_segs)))
        train_segs = set(uniq_segs[:n_seg_train].tolist())
        tr_mask = np.array([o in train_segs for o in owner])
        X_train, X_test = X[tr_mask], X[~tr_mask]
        y_train, y_test = y[tr_mask], y[~tr_mask]
        logging.info(f'按段切分: 训练 {n_seg_train} 段 / 测试 {len(uniq_segs)-n_seg_train} 段')
    else:
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    logging.info(f'训练集: {X_train.shape}，测试集: {X_test.shape}')


    _min   = scaler.min_   if hasattr(scaler, 'min_')   else np.nanmin(data, axis=0)
    _scale = scaler.scale_ if hasattr(scaler, 'scale_') else (
        np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    data_mean_tf  = tf.constant(_min,    dtype=tf.float32)
    data_scale_tf = tf.constant(_scale,  dtype=tf.float32)
    t_std_tf      = tf.constant(86400.0, dtype=tf.float32)  # 1 天 = 86400 s


    feat_dim  = X.shape[-1]
    model     = build_pinn_lstm(args.look_back, feat_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.summary(print_fn=lambda x: logging.info(x))

    train_step_fn = make_train_step(
        model, optimizer, data_scale_tf, data_mean_tf,
        t_std_tf, args.pinn_weight
    )

    logging.info(
        f'开始 PI-LSTM 训练: epochs={args.epochs}, '
        f'batch={args.batch_size}, pinn_weight={args.pinn_weight}'
    )

    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=42)
                .batch(args.batch_size))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
              .batch(args.batch_size)) if len(X_test) > 0 else None

    log_every = 1 if args.epochs <= 20 else max(1, args.epochs // 10)

    best_val   = float('inf')
    best_w     = None
    patience   = max(5, args.epochs // 5)
    bad_epochs = 0

    for epoch in range(args.epochs):
        e_total = e_data = e_phys = 0.0
        n_steps = 0
        for xb, yb in train_ds:
            t_l, d_l, p_l = train_step_fn(xb, yb)
            e_total += float(t_l)
            e_data  += float(d_l)
            e_phys  += float(p_l)
            n_steps += 1

        val_loss = None
        if val_ds is not None:
            v_sum = 0.0; v_n = 0
            for xb, yb in val_ds:
                yp = model(xb, training=False)
                v_sum += float(tf.reduce_mean(tf.square(yp - yb))) * xb.shape[0]
                v_n   += int(xb.shape[0])
            val_loss = v_sum / max(1, v_n)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_w     = model.get_weights()
                bad_epochs = 0
            else:
                bad_epochs += 1

        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            msg = (f'Epoch {epoch+1:>4}/{args.epochs} | '
                   f'Total {e_total/n_steps:.6f} | '
                   f'Data  {e_data/n_steps:.6f} | '
                   f'Phys  {e_phys/n_steps:.6f}')
            if val_loss is not None:
                msg += f' | Val {val_loss:.6f}'
            logging.info(msg)

        if bad_epochs >= patience:
            logging.info(f'Early stop @ epoch {epoch+1} (val 已 {patience} 轮未改善, '
                         f'best val={best_val:.6f})')
            break

    if best_w is not None:
        model.set_weights(best_w)
        logging.info(f'已恢复验证集最佳权重 (val_loss={best_val:.6f})')

    ckpt_path = os.path.join(args.log_dir, f'model_{ts}.weights.h5')
    try:
        model.save_weights(ckpt_path)
        logging.info(f'模型权重已保存: {ckpt_path}')
    except Exception as e:
        logging.warning(f'权重保存失败: {e}')

    if args.continual:
        logging.info('=' * 60)
        logging.info('进入持续学习阶段: 物理模拟伪标签 + Replay/EWC')
        logging.info(f'  风场来源: {"真实 NC: " + args.wind_nc_dir if args.wind_nc_dir else "合成风场"}')
        logging.info(f'  模拟年份: {args.sim_years} | 每年样本: {args.samples_per_year}')
        logging.info(f'  replay_ratio={args.replay_ratio} | ewc_lambda={args.ewc_lambda}')
        logging.info('=' * 60)

        if feat_dim < 4:
            logging.warning('特征维度<4, 模拟器需要 [u,v,A,h,depth] 5 维; 将用占位补齐。')

        start_points = make_start_points_from_history(data, dates)

        sim_by_year = simulate_free_drift_pseudolabels(
            start_points=start_points,
            sim_years=args.sim_years,
            wind_nc_dir=args.wind_nc_dir,
            integrate_hours=args.integrate_hours,
            substep_hours=args.substep_hours,
            samples_per_year=args.samples_per_year,
            seed=42,
        )

        if feat_dim != 5:
            for yr in list(sim_by_year.keys()):
                f, d = sim_by_year[yr]
                sim_by_year[yr] = (f[:, :feat_dim], d)

        cl_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr * 0.1)
        logging.info(f'[CL] 使用独立 optimizer, lr={args.lr * 0.1:.2e}')

        model = continual_learning_loop(
            model=model, optimizer=cl_optimizer,
            old_scaled=scaled, sim_by_year=sim_by_year,
            scaler=scaler, look_back=args.look_back, feat_dim=feat_dim,
            data_scale_tf=data_scale_tf, data_mean_tf=data_mean_tf, t_std_tf=t_std_tf,
            pinn_weight=args.pinn_weight,
            replay_ratio=args.replay_ratio, ewc_lambda=args.ewc_lambda,
            epochs_per_year=args.epochs_per_year, batch_size=args.batch_size, seed=42,
        )

        ckpt_cl = os.path.join(args.log_dir, f'model_continual_{ts}.weights.h5')
        try:
            model.save_weights(ckpt_cl)
            logging.info(f'持续学习后模型权重已保存: {ckpt_cl}')
        except Exception as e:
            logging.warning(f'CL 权重保存失败: {e}')

    if args.weak_label:
        logging.info('=' * 60)
        logging.info('进入弱标签微调阶段: 低分辨率漂移产品 + 加权 Replay/EWC')
        logging.info(f'  漂移产品目录: {args.drift_nc_dir}')
        logging.info(f'  单位换算系数: {args.drift_unit_to_ms:.6g} (->m/s)')
        logging.info('=' * 60)

        weak_by_year = load_drift_weak_labels(
            drift_nc_dir=args.drift_nc_dir,
            sim_years=args.sim_years,
            target_lat=target_lat, target_lon=target_lon,
            unit_to_ms=args.drift_unit_to_ms,
            feat_dim=feat_dim,
        )

        if not weak_by_year:
            logging.warning('未加载到任何弱标签, 跳过弱标签微调。'
                            '请检查 --drift_nc_dir 与产品变量名/单位。')
        else:

            wl_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr * 0.1)
            logging.info(f'[弱标签] 使用独立 optimizer, lr={args.lr * 0.1:.2e}')

            model = weak_label_finetune_loop(
                model=model, optimizer=wl_optimizer,
                old_scaled=scaled, weak_by_year=weak_by_year,
                scaler=scaler, look_back=args.look_back, feat_dim=feat_dim,
                data_scale_tf=data_scale_tf, data_mean_tf=data_mean_tf, t_std_tf=t_std_tf,
                pinn_weight=args.pinn_weight,
                replay_ratio=args.replay_ratio, ewc_lambda=args.ewc_lambda,
                epochs_per_year=args.epochs_per_year, batch_size=args.batch_size, seed=42,
            )
            ckpt_wl = os.path.join(args.log_dir, f'model_weaklabel_{ts}.weights.h5')
            try:
                model.save_weights(ckpt_wl)
                logging.info(f'弱标签微调后模型权重已保存: {ckpt_wl}')
            except Exception as e:
                logging.warning(f'弱标签权重保存失败: {e}')


    ensemble_models = None
    if args.uncertainty == 'ensemble':
        logging.info('=' * 60)
        logging.info(f'训练 Deep Ensemble ({args.ensemble_size} 个模型) 用于不确定性估计')
        logging.info('=' * 60)
        ensemble_models = [model] 
        for m in range(args.ensemble_size - 1):
            tf.random.set_seed(1000 + m)
            mdl_m = build_pinn_lstm(args.look_back, feat_dim)
            opt_m = tf.keras.optimizers.Adam(learning_rate=args.lr)
            step_m = make_train_step(mdl_m, opt_m, data_scale_tf, data_mean_tf,
                                     t_std_tf, args.pinn_weight)
            ds_m = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    .shuffle(len(X_train), seed=1000 + m).batch(args.batch_size))
            for _ in range(args.epochs):
                for xb, yb in ds_m:
                    step_m(xb, yb)
            ensemble_models.append(mdl_m)
            logging.info(f'  Ensemble 成员 {m + 2}/{args.ensemble_size} 训练完成')

    phys_min = np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0][:feat_dim], dtype=np.float32)
    phys_max = np.array([ np.inf,  np.inf, 1.0, np.inf, np.inf][:feat_dim], dtype=np.float32)

    if seg_ids is not None and len(seg_ids) == len(scaled):
        last_seg = seg_ids[-1]
        seg_mask = (seg_ids == last_seg)
        seg_rows = np.where(seg_mask)[0]
        if len(seg_rows) >= args.look_back:
            seed_idx = seg_rows[-args.look_back:]
        else:
            logging.warning(f'末段长度 {len(seg_rows)} < look_back {args.look_back}, '
                            f'预测种子退回全局尾部(可能跨段)。')
            seed_idx = np.arange(len(scaled) - args.look_back, len(scaled))
    else:
        seed_idx = np.arange(len(scaled) - args.look_back, len(scaled))
    seed_scaled = scaled[seed_idx].copy()

    static_depth = seed_scaled[-1, 4] if feat_dim >= 5 else None  # 深度是静态场, 不可外推

    seq   = seed_scaled.copy()
    preds = []
    for _ in range(predict_steps):
        inp = seq.reshape(1, args.look_back, feat_dim)
        p   = model(inp, training=False).numpy()[0]

        p_phys = scaler.inverse_transform(p.reshape(1, -1))[0]
        p_phys = np.clip(p_phys, phys_min, phys_max)
        p = scaler.transform(p_phys.reshape(1, -1))[0].astype(np.float32)

        if static_depth is not None:
            p[4] = static_depth

        preds.append(p)
        seq = np.vstack([seq[1:], p])

    preds_inv = scaler.inverse_transform(np.vstack(preds))
    preds_inv = np.clip(preds_inv, phys_min, phys_max)

    pred_std = None
    if args.uncertainty != 'none':
        seq_init = seed_scaled.copy()
        if args.uncertainty == 'mc_dropout':
            logging.info(f'MC Dropout 不确定性估计 ({args.mc_samples} 次采样)...')
            mc_mean, pred_std = predict_mc_dropout(
                model, seq_init, scaler, args.look_back, feat_dim,
                predict_steps, mc_samples=args.mc_samples)
            preds_inv = mc_mean  # 用 MC 均值作为点预测
        elif args.uncertainty == 'ensemble' and ensemble_models:
            logging.info(f'Deep Ensemble 不确定性估计 ({len(ensemble_models)} 模型)...')
            en_mean, pred_std = predict_ensemble(
                ensemble_models, seq_init, scaler, args.look_back, feat_dim, predict_steps)
            preds_inv = en_mean

        if pred_std is not None:
            ci_lower = preds_inv - 1.96 * pred_std
            ci_upper = preds_inv + 1.96 * pred_std
            logging.info('预测不确定性 (u/v 分量 95% 置信区间, m/s):')
            for t in range(predict_steps):
                logging.info(f'  步 {t+1}: u={preds_inv[t,0]:.4f} '
                             f'[{ci_lower[t,0]:.4f}, {ci_upper[t,0]:.4f}] | '
                             f'v={preds_inv[t,1]:.4f} '
                             f'[{ci_lower[t,1]:.4f}, {ci_upper[t,1]:.4f}]')

        mu, inv_cov = fit_train_distribution(scaled[:split + args.look_back])
        ood_flags = assess_ood(seq_init, mu, inv_cov, threshold=args.ood_threshold)
        n_ood = sum(1 for _, _, f in ood_flags if f)
        max_d = max(d for _, d, _ in ood_flags)
        if n_ood > 0:
            logging.warning(f'[OOD 预警] 输入窗口中 {n_ood}/{len(ood_flags)} 步偏离训练分布 '
                            f'(最大马氏距离 {max_d:.2f} > 阈值 {args.ood_threshold})。'
                            f'此情形 (如极端风速/低密集度) 下预测可信度降低, 请警惕。')
        else:
            logging.info(f'[OOD 检查] 输入窗口在训练分布内 (最大马氏距离 {max_d:.2f})。')

    last_date_str = dates[-1]
    dates_pred    = []
    for fmt in ('%Y-%m-%d %H:%M', '%Y%m%d'):
        try:
            last_date  = datetime.strptime(last_date_str, fmt)
            dates_pred = [(last_date + timedelta(days=i + 1)).strftime(fmt)
                          for i in range(predict_steps)]
            break
        except ValueError:
            continue
    if not dates_pred:
        dates_pred = [f'Step_{i+1}' for i in range(predict_steps)]

    if target_lat is not None and target_lon is not None:
        print_prediction_result(preds_inv, dates_pred, target_lon, target_lat)
    else:
        cols = ['u', 'v', 'A', 'h', 'depth'][:feat_dim]
        print("\n==================== 预测结果（区域平均）====================")
        print(f"{'日期':<22}" + "".join(f"{c:<12}" for c in cols))
        print("-" * (22 + 12 * feat_dim))
        for pred, dt in zip(preds_inv, dates_pred):
            print(f"{dt:<22}" + "".join(f"{v:<12.5f}" for v in pred[:feat_dim]))

    if seg_ids is not None and len(seg_ids) == len(data):
        _last_rows = np.where(seg_ids == seg_ids[-1])[0]
        data_for_export = data[_last_rows[-100:]] if len(_last_rows) else data[-100:]
    else:
        data_for_export = data[-100:]
    export_json(data_for_export, preds_inv, dates_pred, args.log_dir, ts, predict_steps, pred_std=pred_std)

    if data.shape[1] >= 2:
        import matplotlib
        matplotlib.rcParams['axes.unicode_minus'] = False
        feat_names = ['u (East, m/s)', 'v (North, m/s)',
                      'A (Concentration)', 'h (Thickness, m)'][:feat_dim]
        n_hist = min(60, len(data))

        if seg_ids is not None and len(seg_ids) == len(data):
            last_seg_rows = np.where(seg_ids == seg_ids[-1])[0]
            hist = data[last_seg_rows[-n_hist:]] if len(last_seg_rows) >= 1 else data[-n_hist:]
            n_hist = len(hist)
        else:
            hist = data[-n_hist:]
        hist_x = list(range(-n_hist + 1, 1))            
        pred_x = list(range(1, predict_steps + 1))       

        ncols = 2
        nrows = (feat_dim + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
        axes = np.array(axes).reshape(-1)

        for k in range(feat_dim):
            ax = axes[k]
            ax.plot(hist_x, hist[:, k], '-o', color='steelblue',
                    markersize=3, linewidth=1.1, label='History')
            ax.plot([0] + pred_x, [hist[-1, k]] + list(preds_inv[:, k]),
                    '-x', color='crimson', linewidth=1.8, markersize=6,
                    label=f'Prediction ({predict_steps} steps)')
            if pred_std is not None:
                lo = preds_inv[:, k] - 1.96 * pred_std[:, k]
                hi = preds_inv[:, k] + 1.96 * pred_std[:, k]
                ax.fill_between(pred_x, lo, hi, color='crimson', alpha=0.18,
                                label='95% CI')
            ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
            ax.set_xlabel('Step (relative to now)')
            ax.set_ylabel(feat_names[k])
            ax.set_title(feat_names[k])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)

        for k in range(feat_dim, len(axes)):
            axes[k].axis('off')

        fig.suptitle('PI-LSTM Sea Ice Drift Prediction', fontsize=13)
        plt.tight_layout()

        if args.no_plot:
            fig_path = os.path.join(args.log_dir, f'prediction_{ts}.png')
            plt.savefig(fig_path, dpi=150)
            logging.info(f'图表已保存: {fig_path}')
            plt.close()
        else:
            try:
                plt.show()
            except KeyboardInterrupt:
                logging.info('图表显示被中断，继续运行。')
            finally:
                plt.close()

    logging.info('运行完毕。')
