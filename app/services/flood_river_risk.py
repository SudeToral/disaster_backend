# app/services/flood_river_risk.py

import os
import json
from typing import Dict, Any, List

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa
from rasterstats import zonal_stats


# =============================================================================
# CONFIG (edit these paths for your server)
# =============================================================================
DATA_DIR = os.getenv("GLOFAS_DATA_DIR", "data/glofas")

# Forecast netcdf produced by CDS (your downloaded file)
FORECAST_NC = os.getenv("GLOFAS_FORECAST_NC", os.path.join(DATA_DIR, "data_operational-version-4.nc"))

# Turkey admin boundaries (you already have this)
GADM_PATH = os.getenv("TURKEY_GADM_PATH", "data/glofas/gadm41_TUR_2.json")
NAME_COL = os.getenv("TURKEY_NAME_COL", "NAME_1")

# Threshold files (ONLY up to 10y)
THRESH_DIR = os.getenv("GLOFAS_THRESH_DIR", os.path.join(DATA_DIR, "thresholds"))
THRESH_FILES = {
    2: os.getenv("GLOFAS_RL2", "flood_threshold_glofas_v4_rl_2.0.nc"),
    5: os.getenv("GLOFAS_RL5", "flood_threshold_glofas_v4_rl_5.0.nc"),
    10: os.getenv("GLOFAS_RL10", "flood_threshold_glofas_v4_rl_10.0.nc"),
}

# If at least this fraction of river pixels in a province exceed RP => rp_area=RP
AREA_FRAC_GATE = float(os.getenv("GLOFAS_AREA_FRAC_GATE", "0.05"))

# Raster nodata used for zonal_stats
NODATA = float(os.getenv("GLOFAS_NODATA", "-999.0"))

# How many provinces to return/print
TOP_N = int(os.getenv("GLOFAS_TOP_N", "81"))

# Debug printing to console
DEBUG_PRINT = os.getenv("GLOFAS_DEBUG_PRINT", "true").lower() in ("1", "true", "yes", "y")


# =============================================================================
# Helpers
# =============================================================================
def _ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def _period_label(td64) -> str:
    td = pd.to_timedelta(td64)
    days = int(td.total_seconds() // 86400)
    return f"{days} days"


def _open_threshold_da(rp: int) -> xr.DataArray:
    """Open threshold dataset and return the first DataArray, normalized to (latitude, longitude)."""
    fpath = os.path.join(THRESH_DIR, THRESH_FILES[rp])
    _ensure_exists(fpath)
    ds = xr.open_dataset(fpath, decode_cf=False)



    vname = list(ds.data_vars.keys())[0]
    da = ds[vname]

    # normalize coord names if needed
    if "lat" in da.dims and "lon" in da.dims:
        da = da.rename({"lat": "latitude", "lon": "longitude"})
    return da


def _interp_to_forecast_grid(th_da: xr.DataArray, fc_da: xr.DataArray) -> xr.DataArray:
    """Crop and interpolate threshold DA onto forecast grid."""
    lat_min = float(fc_da["latitude"].min())
    lat_max = float(fc_da["latitude"].max())
    lon_min = float(fc_da["longitude"].min())
    lon_max = float(fc_da["longitude"].max())

    # many datasets have latitude descending
    th_crop = th_da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    th_i = th_crop.interp(
        latitude=fc_da["latitude"],
        longitude=fc_da["longitude"],
        method="linear",
    )
    return th_i


def _severity_from_rp_area(rp_area: int) -> str:
    if rp_area >= 10:
        return "MAJOR (>=10y)"
    if rp_area >= 5:
        return "MODERATE (>=5y)"
    if rp_area >= 2:
        return "MINOR (>=2y)"
    return "NORMAL (<2y)"


def _actions_for_severity(sev: str) -> List[str]:
    if sev.startswith("MAJOR"):
        return [
            "Activate municipal/AFAD flood readiness; pre-position pumps, sandbags, mobile barriers.",
            "Inspect & clear storm drains/culverts/underpasses within 12 hours.",
            "Warn river-adjacent neighborhoods; restrict low crossings if needed.",
            "Increase monitoring frequency (hourly).",
        ]
    if sev.startswith("MODERATE"):
        return [
            "Increase monitoring; check known choke points (bridges/culverts).",
            "Prepare temporary barriers and ensure emergency equipment readiness.",
            "Send targeted alerts for basements/underpasses/river-adjacent zones.",
        ]
    if sev.startswith("MINOR"):
        return [
            "Routine monitoring; brief local units.",
            "Be ready to escalate if exceedance expands.",
        ]
    return [
        "No immediate action beyond routine monitoring.",
    ]


def _build_why(row: pd.Series, rps_sorted: List[int]) -> List[str]:
    reasons = []
    reasons.append(f"Peak expected at lead {row['peak_period']} (valid_end={row['valid_end']}).")
    reasons.append(f"Return-period exceedance: rp_area={int(row['rp_area'])}y, rp_any={int(row['rp_any'])}y.")
    for rp in rps_sorted:
        frac = row.get(f"exceed_frac_rp{rp}")
        if frac is not None and np.isfinite(frac):
            reasons.append(f"Fraction of river pixels exceeding {rp}y threshold: {frac*100:.1f}%.")
    if int(row["rp_any"]) >= 10 and int(row["rp_area"]) < 5:
        reasons.append("Hotspot-like exceedance: small affected river area but high local exceedance.")
    return reasons


def _zonal_stats_mean_max(
    gdf: gpd.GeoDataFrame,
    raster_arr: np.ndarray,
    transform,
    name_col_values: np.ndarray,
) -> pd.DataFrame:
    stats = zonal_stats(
        vectors=gdf,
        raster=raster_arr,
        affine=transform,
        stats=["mean", "max"],
        nodata=NODATA,
    )
    df = pd.DataFrame(stats)
    df["province"] = name_col_values
    prov = (
        df.groupby("province")
        .agg({"max": "max", "mean": "mean"})
        .reset_index()
    )
    return prov


def _zonal_stats_mean(
    gdf: gpd.GeoDataFrame,
    raster_arr: np.ndarray,
    transform,
    name_col_values: np.ndarray,
    out_col: str,
) -> pd.DataFrame:
    stats = zonal_stats(
        vectors=gdf,
        raster=raster_arr,
        affine=transform,
        stats=["mean"],
        nodata=NODATA,
    )
    df = pd.DataFrame(stats)
    df["province"] = name_col_values
    prov = (
        df.groupby("province")["mean"]
        .mean()
        .reset_index()
        .rename(columns={"mean": out_col})
    )
    return prov


# =============================================================================
# Main entry point
# =============================================================================
def run_river_flood_risk() -> Dict[str, Any]:
    """
    Runs GloFAS-based river flood risk assessment for Turkey provinces:
    - Reads forecast discharge (dis24) for 1/2/3 days
    - Masks to river pixels using rl_2 validity
    - Computes per-province max/mean discharge
    - Computes exceedance fractions for rl_2/5/10
    - Aggregates 72h peak and derives rp_any / rp_area / severity
    Returns a JSON-serializable payload.
    """

    # ---- validate inputs
    _ensure_exists(FORECAST_NC)
    _ensure_exists(GADM_PATH)
    for rp, fname in THRESH_FILES.items():
        _ensure_exists(os.path.join(THRESH_DIR, fname))

    # ---- load forecast
    ds = xr.open_dataset(FORECAST_NC, decode_cf=False)

    if "dis24" not in ds:
        raise ValueError(f"'dis24' not found in forecast file. Vars: {list(ds.data_vars)}")
    # forecast_reference_time -> datetime
    frt_var = ds["forecast_reference_time"]
    frt_val = frt_var.values[0]

    # 1) Eğer zaten datetime64 ise
    if np.issubdtype(np.array(frt_val).dtype, np.datetime64):
        run_time = pd.to_datetime(frt_val)

    # 2) Numeric + units (e.g., "seconds since 1970-01-01 00:00:00")
    else:
        units = (frt_var.attrs.get("units") or "").lower()

        # defaults
        base = pd.Timestamp("1970-01-01 00:00:00")

        # try to parse base time from units
        # examples: "seconds since 1970-01-01 00:00:00"
        if "since" in units:
            try:
                base_str = units.split("since", 1)[1].strip()
                base = pd.to_datetime(base_str)
            except Exception:
                pass

        v = float(frt_val)

        if units.startswith("second"):
            run_time = base + pd.to_timedelta(v, unit="s")
        elif units.startswith("hour"):
            run_time = base + pd.to_timedelta(v, unit="h")
        elif units.startswith("day"):
            run_time = base + pd.to_timedelta(v, unit="D")
        else:
            # fallback: seconds
            run_time = base + pd.to_timedelta(v, unit="s")


    # forecast_period -> timedelta64
    fp = ds["forecast_period"]

    # netcdf'lerde bazen fp numeric + units olur (örn: "days" veya "hours")
    units = (fp.attrs.get("units") or "").lower()

    vals = fp.values
    # scalar değilse dizi
    if np.issubdtype(vals.dtype, np.number):
        if "day" in units:
            periods = np.array([np.timedelta64(int(v), "D") for v in vals])
        elif "hour" in units:
            periods = np.array([np.timedelta64(int(v), "h") for v in vals])
        elif "second" in units:
            periods = np.array([np.timedelta64(int(v), "s") for v in vals])
        else:
            # fallback: GloFAS çoğunlukla days
            periods = np.array([np.timedelta64(int(v), "D") for v in vals])
    else:
        # zaten timedelta gibi geldiyse
        periods = vals


    if DEBUG_PRINT:
        print("Run time:", run_time)
        print("Forecast periods:", [_period_label(p) for p in periods])

    # ---- load shapes
    gdf = gpd.read_file(GADM_PATH).to_crs("EPSG:4326")
    name_vals = gdf[NAME_COL].values

    # ---- reference grid (for threshold interpolation)
    fc_ref = ds["dis24"].isel(forecast_period=0, forecast_reference_time=0)

    # ---- load + interp thresholds (2/5/10)
    rps_sorted = sorted(THRESH_FILES.keys())
    th_interp: Dict[int, xr.DataArray] = {}
    for rp in rps_sorted:
        th = _open_threshold_da(rp)
        th_interp[rp] = _interp_to_forecast_grid(th, fc_ref)

    # ---- river mask from rl_2 (valid and >0)
    river_mask = np.isfinite(th_interp[2].values) & (th_interp[2].values > 0)

    # ---- per period province tables
    tables = []

    for i in range(ds.sizes["forecast_period"]):
        fp = periods[i]
        fp_label = _period_label(fp)

        da = ds["dis24"].isel(forecast_period=i, forecast_reference_time=0)
        dis = da.values.astype("float32")

        # mask non-river pixels for discharge stats
        dis_masked = np.where(river_mask, dis, np.nan)

        # build DataArray for zonal_stats (needs x/y + transform)
        da2 = xr.DataArray(
            np.where(np.isfinite(dis_masked), dis_masked, NODATA),
            coords={"latitude": da["latitude"], "longitude": da["longitude"]},
            dims=("latitude", "longitude"),
            name="dis24_river",
        )
        da2 = (
            da2.rename({"longitude": "x", "latitude": "y"})
            .rio.write_crs("EPSG:4326")
            .rio.write_nodata(NODATA)
        )
        transform = da2.rio.transform()

        prov_dis = _zonal_stats_mean_max(gdf, np.asarray(da2.values), transform, name_vals)
        prov_dis = prov_dis.rename(columns={"max": "max_discharge", "mean": "mean_discharge"})
        prov_dis["period"] = fp_label
        prov_dis["valid_end"] = (run_time + pd.to_timedelta(fp)).date()

        # exceedance fractions for each rp
        for rp, th_da in th_interp.items():
            thr = th_da.values.astype("float32")
            exc = (dis > thr) & river_mask & np.isfinite(thr)
            exc_arr = np.where(river_mask, exc.astype("float32"), NODATA)

            exc_da = xr.DataArray(
                exc_arr,
                coords={"latitude": da["latitude"], "longitude": da["longitude"]},
                dims=("latitude", "longitude"),
                name=f"exc_rp{rp}",
            )
            exc_da = (
                exc_da.rename({"longitude": "x", "latitude": "y"})
                .rio.write_crs("EPSG:4326")
                .rio.write_nodata(NODATA)
            )

            prov_exc = _zonal_stats_mean(
                gdf,
                np.asarray(exc_da.values),
                exc_da.rio.transform(),
                name_vals,
                out_col=f"exceed_frac_rp{rp}",
            )
            prov_dis = prov_dis.merge(prov_exc, on="province", how="left")

        tables.append(prov_dis)

    allp = pd.concat(tables, ignore_index=True)

    # ---- debug prints: top N per lead time
    if DEBUG_PRINT:
        for d in ["1 days", "2 days", "3 days"]:
            print("\n====================")
            print(f"TOP {TOP_N} | {d} (dis24, river-masked)")
            print("====================")
            tmp = (
                allp[allp["period"] == d]
                .sort_values("max_discharge", ascending=False)
                .head(TOP_N)
            )
            print(tmp[["province", "max_discharge", "mean_discharge", "valid_end"]])

    # ---- 72h peak per province
    idx = allp.groupby("province")["max_discharge"].idxmax()
    peak = allp.loc[idx].copy()
    peak = peak.rename(
        columns={
            "max_discharge": "peak_72h_max_discharge",
            "mean_discharge": "mean_at_peak",
            "period": "peak_period",
        }
    )
    peak = peak.sort_values("peak_72h_max_discharge", ascending=False)

    def compute_rp_any(row: pd.Series) -> int:
        rp_any = 0
        for rp in rps_sorted:
            frac = row.get(f"exceed_frac_rp{rp}")
            if frac is not None and np.isfinite(frac) and frac > 0:
                rp_any = rp
        return int(rp_any)

    def compute_rp_area(row: pd.Series) -> int:
        rp_area = 0
        for rp in rps_sorted:
            frac = row.get(f"exceed_frac_rp{rp}")
            if frac is not None and np.isfinite(frac) and frac >= AREA_FRAC_GATE:
                rp_area = rp
        return int(rp_area)

    peak["rp_any"] = peak.apply(compute_rp_any, axis=1)
    peak["rp_area"] = peak.apply(compute_rp_area, axis=1)
    peak["severity"] = peak["rp_area"].apply(lambda x: _severity_from_rp_area(int(x)))
    peak["why"] = peak.apply(lambda r: _build_why(r, rps_sorted), axis=1)
    peak["actions"] = peak["severity"].apply(_actions_for_severity)

    # ---- debug: risk report top N
    if DEBUG_PRINT:
        print("\n======================================================================")
        print(f"TOP {TOP_N} | 72h WINDOW RETURN-PERIOD RISK REPORT (<=10y)")
        print("======================================================================\n")
        top = peak.head(TOP_N)
        for _, r in top.iterrows():
            print("-" * 70)
            print(f"{r['province']} | severity={r['severity']}")
            print(
                f"peak_72h_max={r['peak_72h_max_discharge']:.1f} m3/s at {r['peak_period']} "
                f"(valid_end={r['valid_end']})"
            )
            print(f"rp_area={int(r['rp_area'])}y | rp_any={int(r['rp_any'])}y")
            print("WHY:")
            for x in r["why"]:
                print(f" - {x}")
            print("ACTIONS:")
            for a in r["actions"]:
                print(f" - {a}")
            print()

    # ---- build JSON payload (top N)
    top = peak.head(TOP_N)

    payload: Dict[str, Any] = {
        "forecast_reference_time": str(run_time),
        "window_hours": 72,
        "thresholds_used": rps_sorted,
        "area_fraction_gate": AREA_FRAC_GATE,
        "top_provinces": [],
    }

    for _, r in top.iterrows():
        item = {
            "province": r["province"],
            "severity": r["severity"],
            "valid_end": str(r["valid_end"]),
            "why": r["why"],
            "actions": r["actions"],
            "metrics": {
                "peak_72h_discharge_max": float(r["peak_72h_max_discharge"]),
                "peak_period": r["peak_period"],
                "rp_area_years": int(r["rp_area"]),
                "rp_any_years": int(r["rp_any"]),
            },
        }
        for rp in rps_sorted:
            col = f"exceed_frac_rp{rp}"
            if col in r and pd.notna(r[col]):
                item["metrics"][col] = float(r[col])

        payload["top_provinces"].append(item)

    return payload


# Optional CLI run for local debugging:
if __name__ == "__main__":
    out = run_river_flood_risk()
    print("\n======================================================================")
    print("JSON PAYLOAD (TOP N)")
    print("======================================================================")
    print(json.dumps(out, ensure_ascii=False, indent=2))
