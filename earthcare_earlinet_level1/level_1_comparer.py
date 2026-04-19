import os
import sys
import glob
import pandas as pd
import xarray as xr
import re
from .constants import *
from .utilities import *


class EarthCARE_EARLINET_Level1Comparer:

    def __init__(self, files=FILES, folder=FOLDER, folders=FOLDERS, root=ROOT, glob_pattern=GLOB_PATTERN, radii=RADII,
                 gb=GB, gb_lat=GB_LAT, gb_lon=GB_LON, smoothing_window_m=SMOOTH_WIN_M_DEFAULT, save_plots=SAVE_PLOTS,
                 output_directory=OUTPUT_DIRECTORY):
        """
        Initializes the class with specified parameters and processes given folder paths.

        :parameter files: List of paths to files to be processed. Overwrites 'folder' and 'folders' if specified.
                          Should contain a list of tuples, in which the first element is the EarthCARE filepath and the second one is the corresponding EARLINET files.
        :type files: Optional[List[Tuple[Union[str, Path, os.PathLike[str], os.PathLike[os.PathLike[str], Union[str, Path, os.PathLike[str], os.PathLike[os.PathLike[str]]]]]

        :parameter folder: Path to a single folder. Must be an absolute or relative path to one folder or
                           None if using other options.
        :type folder: Optional[str]

        :parameter folders: List of paths to multiple folders. Absolute paths will be derived if not
                            already. Optional, overrides other folder options.
        :type folders: Optional[List[str]]

        :parameter root: Root directory to search for subfolders using a pattern. Must be provided with
                         ``glob_pattern``. An absolute path will be derived if not already.
        :type root: Optional[str]

        :parameter glob_pattern: Pattern to match subfolders within the provided ``root`` directory. Used
                                 to collect directories dynamically if no folder or folders specified.
        :type glob_pattern: Optional[str]

        :parameter radii: Specifies radii, purpose unspecified in provided context but plays a role in
                          logic used within the instance.
        :type radii: Any

        :parameter gb: Which profiles to choose from the ground-based measurements - either a time-constrained window,
                       all the profiles or both, separately
        :type gb: Optional[str], one of ["window", "all", "both"]

        :parameter gb_lat: Ground-based station latitude
        :type gb_lat: float

        :parameter gb_lon: Ground-based station longitude
        :type gb_lon: float

        :parameter smoothing_window_m: Smoothing window for the ground-based measurements, in meters
        :type smoothing_window_m: float

        :parameter save_plots: Whether to save plots to disk
        :type save_plots: bool
        ```"""

        self.files = None

        if files is not None:
            logger.info("Files were specified. Ignoring 'folders' argument.")
            self.files = files
            folders = None

        if folders is not None:
            self.folders = [os.path.abspath(f) for f in folders]
        elif root is not None and glob_pattern is not None:
            root = os.path.abspath(root)

            candidates = glob.glob(os.path.join(root, glob_pattern))
            self.folders = [p for p in candidates if os.path.isdir(p)]
            if not self.folders:
                sys.exit(f"No subfolders found under {root} matching pattern '{glob_pattern}'.")
        elif folder:
            self.folders = [os.path.abspath(folder)]
        else:
            if self.files is not None:
                logger.error(f"Please provide either 'folders' argument or a 'root' and 'glob_pattern' pair.")
        self.radii = radii
        self.gb = gb
        self.gb_lat = gb_lat
        self.gb_lon = gb_lon
        self.smoothing_window_m = smoothing_window_m
        self.save_plots = save_plots
        self.output_directory = output_directory

        self.args = ()

    @staticmethod
    def _get_atlid_time_abs(ds):
        """Return ATLID times as numpy datetime64[s] (absolute)."""
        t = ds["time"].values
        if np.issubdtype(t.dtype, np.datetime64):
            return t.astype("datetime64[s]")
        ref = np.datetime64("2000-01-01T00:00:00")
        return ref + t.astype("timedelta64[s]")

    def atlid_circle_mask(self, ds, radius_km):
        """Mask ATLID along-track samples within `radius_km` of the GB site."""
        lat = ds["sensor_latitude"].values
        lon = ds["sensor_longitude"].values

        try:
            from geopy.distance import geodesic
            d_km = np.array([geodesic((self.gb_lat, self.gb_lon), (lat[i], lon[i])).km for i in range(len(lat))])
        except Exception:
            d_km = haversine_km(self.gb_lat, self.gb_lon, lat, lon)
        mask = d_km <= radius_km
        return mask, lat, lon, d_km

    # Regridding & smoothing
    @staticmethod
    def _get_bin_edges(alt):
        d = np.diff(alt)
        e = np.empty(len(alt)+1, dtype=float)
        e[1:-1] = (alt[:-1] + alt[1:]) / 2.0
        e[0]  = alt[0]  - d[0]  / 2.0
        e[-1] = alt[-1] + d[-1] / 2.0
        return e

    def _regrid_to_target_grid(self, data_fine, alt_fine, alt_target):
        ef, et = self._get_bin_edges(alt_fine), self._get_bin_edges(alt_target)
        out = []
        for y in np.asarray(data_fine):
            yy = np.empty_like(alt_target, dtype=float)
            for i in range(len(alt_target)):
                a, b = et[i], et[i+1]
                overlap = np.minimum(ef[1:], b) - np.maximum(ef[:-1], a)
                overlap = np.clip(overlap, 0, None)
                good = np.isfinite(y) & (overlap > 0)
                yy[i] = np.nan if not np.any(good) else np.sum(y[good]*overlap[good])/np.sum(overlap[good])
            out.append(yy)
        return np.asarray(out)

    @staticmethod
    def _smooth_profiles_2d(P, k):
        P = np.asarray(P, float)
        return np.vstack([moving_nanmean_1d(p, k) for p in P])

    # Quick-looks
    @staticmethod
    def _find_norm_band_robust(alt_m, prof, search_bounds, fallback_band,
                               min_width=1500.0, step_m=50.0):

        dz = float(np.nanmedian(np.diff(alt_m)))
        zmin, zmax = float(np.nanmin(alt_m)), float(np.nanmax(alt_m))
        low_req, high_req = search_bounds
        low  = max(low_req,  zmin + 3*dz)
        high = min(high_req, zmax - 3*dz)
        best, best_score = None, np.inf
        if high - low >= min_width:
            step = max(step_m, max(dz, 25.0))
            starts = np.arange(low, high - min_width, step)
            for start in starts:
                end = start + min_width
                m = (alt_m >= start) & (alt_m <= end)
                vals = prof[m]
                if vals.size < 5:
                    continue
                mu = np.nanmean(vals); sd = np.nanstd(vals)
                score = sd + 3.0*max(mu - 2e-6, 0.0)
                if score < best_score:
                    best_score, best = score, (float(start), float(end))
        if best is None:
            f0, f1 = fallback_band
            f0c = max(f0, zmin + 3*dz); f1c = min(f1, zmax - 3*dz)
            if (f1c - f0c) >= min_width:
                best = (float(f0c), float(f0c + min_width))
            else:
                best = (max(zmin, zmax - min_width), float(zmax))
        return best

    def quicklook_compute(self, meas_2d, alt_meas, mol_2d, alt_mol,
                          smooth_win_m=SMOOTH_WIN_M_DEFAULT,
                          search_bounds=(8000.0, 11000.0),
                          fallback_band=(8000.0, 11000.0),
                          min_width_m=1500.0):
        """
        Inputs:
          meas_2d : (ntime, nz_meas) measured attenuated backscatter
          alt_meas: (nz_meas,) meters, increasing
          mol_2d  : (ntime, nz_mol) molecular attenuated backscatter
          alt_mol : (nz_mol,) meters
        Returns dict with:
          alt, meas (scaled), meas_sd, mol, reldiff, reldiff_sd, band, rsem
        """

        if (len(alt_mol) != len(alt_meas) or
            np.nanmax(np.abs(np.diff(alt_mol) - np.diff(alt_meas))) > 1e-6):
            mol_on_meas = self._regrid_to_target_grid(mol_2d, alt_mol, alt_meas)
        else:
            mol_on_meas = np.asarray(mol_2d, float)

        # vertical smoothing
        dz = float(np.nanmedian(np.diff(alt_meas)))
        k  = max(1, int(round(smooth_win_m / max(dz, 1))))
        meas_s = self._smooth_profiles_2d(np.asarray(meas_2d, float), k)
        mol_s  = self._smooth_profiles_2d(np.asarray(mol_on_meas, float), k)

        # time-median profiles and global scale in a clean band
        meas_mu = np.nanmedian(meas_s, axis=0)
        meas_sd = np.nanstd   (meas_s, axis=0)
        mol_mu  = np.nanmedian(mol_s,  axis=0)

        band   = self._find_norm_band_robust(alt_meas, meas_mu, search_bounds, fallback_band,
                                             min_width=min_width_m, step_m=int(max(dz, 50)))
        z0, z1 = band
        m_norm = (alt_meas >= z0) & (alt_meas <= z1)

        scale = np.nanmean(mol_mu[m_norm]) / np.nanmean(meas_mu[m_norm])
        meas_scaled     = meas_mu * scale
        meas_scaled_sd  = meas_sd  * scale

        mol_floor = np.nanpercentile(mol_mu[np.isfinite(mol_mu)], 5) if np.isfinite(mol_mu).any() else 1e-12
        reldiff     = (meas_scaled - mol_mu) / mol_mu
        reldiff_sd  = meas_scaled_sd / np.maximum(mol_mu, mol_floor)
        rsem        = float(np.nanstd(meas_scaled[m_norm]) / np.nanmean(meas_scaled[m_norm]))

        return dict(alt=alt_meas, meas=meas_scaled, meas_sd=meas_scaled_sd,
                    mol=mol_mu, reldiff=reldiff, reldiff_sd=reldiff_sd,
                    band=(float(z0), float(z1)), rsem=rsem)


    # Agreement metrics
    @staticmethod
    def _align_by_lag(x, y, max_shift_bins=1):

        best = (0, -np.inf, x, y)
        for L in range(-max_shift_bins, max_shift_bins+1):
            if L < 0:
                xs, ys = x[-L:], y[:len(y)+L]
            elif L > 0:
                xs, ys = x[:-L], y[L:]
            else:
                xs, ys = x, y
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() >= 3:
                r = np.corrcoef(xs[m], ys[m])[0,1]
                if np.isfinite(r) and r > best[1]:
                    best = (L, r, xs[m], ys[m])
        return best

    def band_metrics(self, alt_m, sr_atl, sr_gb, sd_atl=None, sd_gb=None,
                     band_km=(1,4), max_lag_bins=1, delta=0.05):

        z0, z1 = band_km[0]*1000.0, band_km[1]*1000.0
        m = (alt_m >= z0) & (alt_m < z1) & np.isfinite(sr_atl) & np.isfinite(sr_gb)
        if m.sum() < 3:
            return dict(band=band_km, N=int(m.sum()), lag_bins=np.nan,
                        CCC=np.nan, bias=np.nan, MAE=np.nan, RMSE=np.nan,
                        wRMSE=np.nan, red_chi2=np.nan, tost_pass=False,
                        ci_lo=np.nan, ci_hi=np.nan, delta=float(delta))

        xa, ya = sr_atl[m].astype(float), sr_gb[m].astype(float)
        L, _, xL, yL = self._align_by_lag(xa, ya, max_shift_bins=max_lag_bins)

        if sd_atl is not None and sd_gb is not None:
            sa = sd_atl[m].astype(float)
            sb = sd_gb[m].astype(float)
            if L < 0:
                saL, sbL = sa[-L:], sb[:len(sb)+L]
            elif L > 0:
                saL, sbL = sa[:-L], sb[L:]
            else:
                saL, sbL = sa, sb
            good = np.isfinite(xL) & np.isfinite(yL) & np.isfinite(saL) & np.isfinite(sbL)
            xL, yL, saL, sbL = xL[good], yL[good], saL[good], sbL[good]
        else:
            saL = sbL = None
            good = np.isfinite(xL) & np.isfinite(yL)
            xL, yL = xL[good], yL[good]

        if xL.size < 3:
            return dict(band=band_km, N=int(xL.size), lag_bins=L,
                        CCC=np.nan, bias=np.nan, MAE=np.nan, RMSE=np.nan,
                        wRMSE=np.nan, red_chi2=np.nan, tost_pass=False,
                        ci_lo=np.nan, ci_hi=np.nan, delta=float(delta))

        d  = xL - yL
        res = {
            "band": band_km,
            "N": int(xL.size),
            "lag_bins": int(L),
            "CCC": float(lin_ccc(xL, yL)),
            "bias": float(np.mean(d)),
            "MAE":  float(np.mean(np.abs(d))),
            "RMSE": float(np.sqrt(np.mean(d**2))),
            "wRMSE": float(weighted_rmse(xL, yL, saL, sbL)),
            "red_chi2": float(reduced_chi2(xL, yL, saL, sbL)),
        }
        ok, ci = tost_equivalence(d, delta=delta)
        res.update({"tost_pass": bool(ok), "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "delta": float(delta)})
        return res


    @staticmethod
    def _plot_sr_case(
        alt_m, atl_sr, atl_sd, gb_sr, gb_sd,
        atl_band=None, gb_band=None, radius_km=50, gb_mode="window",
        out_png=None, closest_km=None, t0=None, t1=None
    ):
        """SR plot (ATLID vs Ground-based on ATLID grid) with poster sizing."""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        alt_km = np.asarray(alt_m, float) / 1000.0
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.plot(atl_sr, alt_km, lw=2, label="ATLID Scattering Ratio")
        if np.isfinite(atl_sd).any():
            ax.fill_betweenx(alt_km, atl_sr - atl_sd, atl_sr + atl_sd, alpha=0.20, lw=0)

        ax.plot(gb_sr, alt_km, lw=2, label="Ground-based Scattering Ratio")
        if np.isfinite(gb_sd).any():
            ax.fill_betweenx(alt_km, gb_sr - gb_sd, gb_sr + gb_sd, alpha=0.20, lw=0)


        ax.axvline(1.0, color="k", lw=1)
        if atl_band is not None:
            ax.axhspan(float(atl_band[0])/1000.0, float(atl_band[1])/1000.0, color="gray", alpha=0.18)
        if gb_band is not None:
            ax.axhspan(float(gb_band[0])/1000.0, float(gb_band[1])/1000.0, color="tan", alpha=0.18)


        ax.set_xlim(0.6, 1.4)
        ax.set_ylim(1, 12)
        ax.set_xlabel("Scattering Ratio", fontsize=LABEL_SIZE)
        ax.set_ylabel("Altitude (km)", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=LEGEND_SIZE)


        def _fmt_time(t):
            try:
                return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                return str(t) if t is not None else "n/a"

        ck = f"{float(closest_km):.1f} km" if closest_km is not None else "n/a"
        t0s = _fmt_time(t0)
        t1s = _fmt_time(t1)

        ax.set_title(
            f"R={radius_km:.0f} km • Ground-based Profile Averaging Mode={gb_mode}\n"
            f"Closest ATLID–GB distance: {ck} \nTime window: {t0s} → {t1s}",
            fontsize=TITLE_SIZE, pad=12
        )

        plt.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=220, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


    @staticmethod
    def plot_sr_and_diff(alt_m,
                         atlid_sr, gb_sr_on_atlid,
                         atlid_sd=None, gb_sd=None,
                         radius_km=50, gb_label="all",
                         save_path=None):
        """
        alt_m            : 1D altitude array (meters) on the ATLID grid
        atlid_sr         : ATLID scattering ratio (mean)
        gb_sr_on_atlid   : GB SR regridded to ATLID grid
        atlid_sd, gb_sd  : 1σ SR uncertainties (same grid), optional
        """
        import numpy as np
        import matplotlib.pyplot as plt

        alt_km = np.asarray(alt_m, float) / 1000.0
        A  = np.asarray(atlid_sr, float)
        G  = np.asarray(gb_sr_on_atlid, float)
        SA = np.asarray(atlid_sd, float) if atlid_sd is not None else None
        SG = np.asarray(gb_sd,  float) if gb_sd  is not None else None


        D = A - G
        if (SA is not None) and (SG is not None):
            SD = np.sqrt(SA**2 + SG**2)
        else:
            SD = None

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(6.6, 8.0), sharey=True,
            gridspec_kw={"height_ratios":[2.2, 1.4], "hspace":0.06}
        )


        ax_top.plot(A, alt_km, lw=2, label="ATLID SR")
        ax_top.plot(G, alt_km, lw=2, label="GB SR (regridded)")
        if SA is not None and np.isfinite(SA).any():
            ax_top.fill_betweenx(alt_km, A-SA, A+SA, alpha=0.20, lw=0)
        if SG is not None and np.isfinite(SG).any():
            ax_top.fill_betweenx(alt_km, G-SG, G+SG, alpha=0.20, lw=0)
        ax_top.axvline(1.0, color="k", lw=1)
        ax_top.set_xlim(0.6, 1.4)
        ax_top.set_ylim(1, 12)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="lower right")
        ax_top.set_title(f"SR • R={int(radius_km)} km • GB={gb_label}")


        ax_bot.plot(D, alt_km, lw=2, label="ATLID − GB")
        if SD is not None and np.isfinite(SD).any():
            ax_bot.fill_betweenx(alt_km, D-SD, D+SD, alpha=0.25, lw=0, label="±1σ")

        ax_bot.axvline(0.0, color="k", lw=1)


        span = np.nanmax(np.abs(D) + (SD if SD is not None else 0))
        if not np.isfinite(span):
            span = 0.15
        span = max(0.15, min(0.5, 1.1*span))
        ax_bot.set_xlim(-span, span)

        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_xlabel("Scattering Ratio  (top)   /   Difference ATLID − GB (bottom)")
        ax_bot.legend(loc="lower right")

        #plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches="tight")
            logger.info(f"[saved] {save_path}")
        plt.show()



    # One-radius

    def _process_one_radius(self, ds_atlid, ds_003, ds_009,
                           radius_km, gb_mode, out_dir,
                           smooth_win_m=SMOOTH_WIN_M_DEFAULT,
                           save_plots=False):
        """
        Compute quick-looks, SR pairing, and band metrics for one circle radius and GB mode.
        Saves:
          - sr_points_{gb_mode}_radius{R}km.csv
          - stats_{gb_mode}_radius{R}km.csv
          - (optional) sr_{gb_mode}_radius{R}km.png
        """

        alpha_mol = ds_003["molecular_extinction"]
        T_det = ds_003["molecular_transmissivity_at_detection_wavelength"]
        T_em  = ds_003["molecular_transmissivity_at_emission_wavelength"]
        beta_mol = alpha_mol / S_MOL
        rayleigh_att = beta_mol * T_det * T_em


        times = pd.to_datetime(ds_009["time"].values).astype("datetime64[ns]")
        alt_rayleigh = ds_003["altitude"].isel(time=0).values
        alt_total_all = ds_009["altitude"].values
        total_signal_all = ds_009["attenuated_backscatter"].sel(channel=0).values


        mask, lat_all, lon_all, dists = self.atlid_circle_mask(ds_atlid, radius_km)
        if mask.sum() < 1:
            logger.warning(f"[radius {radius_km} km] No ATLID profiles inside circle — skipping.")
            return None

        times_atlid_abs = self._get_atlid_time_abs(ds_atlid)
        t0 = times_atlid_abs[mask][0]
        t1 = times_atlid_abs[mask][-1]


        alt_atl = (ds_atlid["sample_altitude"].isel(along_track=0).values
                   - ds_atlid["geoid_offset"].isel(along_track=0).values)
        alt_atl = np.sort(alt_atl[np.isfinite(alt_atl)])


        mie   = ds_atlid["mie_attenuated_backscatter"][mask, :].values
        cross = ds_atlid["crosspolar_attenuated_backscatter"][mask, :].values
        ray   = ds_atlid["rayleigh_attenuated_backscatter"][mask, :].values
        atl_total = mie + cross + ray

        atl = self.quicklook_compute(atl_total, alt_atl, ray, alt_atl,
                                     smooth_win_m=smooth_win_m,
                                     search_bounds=SEARCH_BOUNDS_M_A,
                                     fallback_band=FALLBACK_BAND_M_A)


        if gb_mode == "window":
            gb_time_mask = (times >= t0) & (times <= t1)
            if not np.any(gb_time_mask):
                logger.warning(f"[radius {radius_km} km][{gb_mode}] No GB profiles overlap t0–t1 — skipping.")
                return None
            gb_meas = total_signal_all[gb_time_mask, :]
            gb_mol  = rayleigh_att.sel(channel=0).interp(time=times[gb_time_mask], method="nearest").values
        else:  # "all"
            gb_meas = total_signal_all
            gb_mol  = rayleigh_att.sel(channel=0).interp(time=times, method="nearest").values

        alt_gb  = alt_total_all[0, :]
        gb = self.quicklook_compute(gb_meas, alt_gb, gb_mol, alt_rayleigh,
                                 smooth_win_m=smooth_win_m,
                                 search_bounds=SEARCH_BOUNDS_M_G,
                                 fallback_band=FALLBACK_BAND_M_G)


        atl_sr     = 1.0 + np.asarray(atl["reldiff"], float)
        atl_sd     = np.asarray(atl["reldiff_sd"], float)
        gb_sr_1d   = 1.0 + np.asarray(gb["reldiff"], float)
        gb_sd_1d   = np.asarray(gb["reldiff_sd"], float)

        gb_sr_on_atl = self._regrid_to_target_grid([gb_sr_1d], np.asarray(gb["alt"], float), np.asarray(atl["alt"], float))[0]
        gb_sd_on_atl = self._regrid_to_target_grid([gb_sd_1d], np.asarray(gb["alt"], float), np.asarray(atl["alt"], float))[0]


        closest_distance_km = float(np.nanmin(dists))
        sr_df = pd.DataFrame({
            "altitude_m":          np.asarray(atl["alt"], float),
            "gb_sr_on_atlid":      gb_sr_on_atl,
            "gb_sd_on_atlid":      gb_sd_on_atl,
            "atlid_sr":            atl_sr,
            "atlid_sd":            atl_sd,
            "closest_distance_km": closest_distance_km,
            "circle_radius_km":    float(radius_km),
            "gb_mode":             gb_mode,
            "t0":                  str(t0),
            "t1":                  str(t1),
        })
        out_csv_sr = os.path.join(out_dir, f"sr_points_{gb_mode}_radius{int(radius_km)}km.csv")
        sr_df.to_csv(out_csv_sr, index=False)


        bands = [(1,4), (4,8), (8,12)]
        rows = []
        for b in bands:
            rows.append(self.band_metrics(
                alt_m=np.asarray(atl["alt"], float),
                sr_atl=atl_sr,
                sr_gb=gb_sr_on_atl,
                sd_atl=atl_sd,
                sd_gb=gb_sd_on_atl,
                band_km=b,
                max_lag_bins=(2 if b==(1,4) else 1),
                delta=0.05
            ))
        band_df = pd.DataFrame(rows)
        out_csv_stats = os.path.join(out_dir, f"stats_{gb_mode}_radius{int(radius_km)}km.csv")
        band_df.to_csv(out_csv_stats, index=False)

        if save_plots:
            out_png = os.path.join(out_dir, f"sr_{gb_mode}_radius{int(radius_km)}km.png")
            self._plot_sr_case(
                np.asarray(atl["alt"], float), atl_sr, atl_sd,
                gb_sr_on_atl, gb_sd_on_atl,
                atl_band=atl["band"], gb_band=gb["band"],
                radius_km=radius_km, gb_mode=gb_mode, out_png=out_png,
                closest_km=closest_distance_km,
                t0=t0, t1=t1
            )


        # Console summary
        logger.info(f"\n[{gb_mode}][R={radius_km} km] t0={t0}  t1={t1}")
        logger.info(band_df)

        return dict(sr_csv=out_csv_sr, stats_csv=out_csv_stats)


    # Helper: run a single folder
    def _run_one_folder(self, folder):
        """Discover files in `folder`, run all radii × modes, save outputs."""
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)

        # --- discover files
        files_003 = sorted(glob.glob(os.path.join(folder, "ino_003_*.nc")))
        files_009 = sorted(glob.glob(os.path.join(folder, "ino_009_*.nc")))
        atlid_candidates = glob.glob(os.path.join(folder, "*ATL_NOM_1B_*.h5"))
        if not files_003 or not files_009 or not atlid_candidates:
            logger.warning(f"[skip] {folder} — missing ino_003 / ino_009 / ATLID files")
            return

        file_atlid = atlid_candidates[0]

        # --- load datasets
        logger.info(f"\n=== Folder: {folder} ===")
        logger.info("Loading GB files ...")
        ds_003 = xr.open_mfdataset(files_003, combine="by_coords")
        ds_009 = xr.open_mfdataset(files_009, combine="by_coords")
        logger.info("Loading ATLID ...")
        ds_atlid = xr.open_dataset(file_atlid, group="ScienceData")

        # --- outputs go next to the data
        out_dir = self.output_directory
        os.makedirs(out_dir, exist_ok=True)

        # --- run batch
        modes = ["window", "all"] if self.gb == "both" else [self.gb]
        for R in self.radii:
            for m in modes:
                try:
                    self._process_one_radius(
                        ds_atlid=ds_atlid,
                        ds_003=ds_003,
                        ds_009=ds_009,
                        radius_km=R,
                        gb_mode=m,
                        out_dir=out_dir,
                        smooth_win_m=float(self.smoothing_window_m),
                        save_plots=self.save_plots
                    )
                except Exception as e:
                    logger.error(f"[error] {folder}  R={R} km  GB={m}: {e}")

        # tidy up file handles
        try: ds_003.close()
        except: pass
        try: ds_009.close()
        except: pass
        try: ds_atlid.close()
        except: pass

    def _run_one_instance(self, files):
        """Run one combination of radii × modes, save outputs."""

        # --- discover files
        file_atlid = files[0]
        filename_atlid = os.path.basename(file_atlid)
        ground_based_files = files[1]
        files_003 = sorted([file for file in ground_based_files if re.match("ino_003_.*\.nc", os.path.basename(file))])
        files_009 = sorted([file for file in ground_based_files if re.match("ino_009_.*\.nc", os.path.basename(file))])
        if not files_003 or not files_009 or not file_atlid:
            logger.warning(f"[skip] missing ino_003 / ino_009 / ATLID files - {filename_atlid=}")
            return

        # --- load datasets
        logger.info(f"\n=== File: {file_atlid.split(os.sep)[-1]} ===")
        logger.info("Loading GB files ...")
        ds_003 = xr.open_mfdataset(files_003, combine="by_coords")
        ds_009 = xr.open_mfdataset(files_009, combine="by_coords")
        logger.info("Loading ATLID ...")
        ds_atlid = xr.open_dataset(file_atlid, group="ScienceData")

        # --- outputs go next to the data
        out_dir = os.path.join(self.output_directory, filename_atlid.split(".")[0])
        os.makedirs(out_dir, exist_ok=True)

        # --- run batch
        modes = ["window", "all"] if self.gb == "both" else [self.gb]
        for R in self.radii:
            for m in modes:
                try:
                    self._process_one_radius(
                        ds_atlid=ds_atlid,
                        ds_003=ds_003,
                        ds_009=ds_009,
                        radius_km=R,
                        gb_mode=m,
                        out_dir=out_dir,
                        smooth_win_m=float(self.smoothing_window_m),
                        save_plots=self.save_plots
                    )
                except Exception as e:
                    logger.error(f"[error] R={R} km  GB={m}: {e}")

        # tidy up file handles
        try: ds_003.close()
        except: pass
        try: ds_009.close()
        except: pass
        try: ds_atlid.close()
        except: pass

    def run(self):

        if self.files is not None:
            for f in self.files:
                self._run_one_instance(f)
            logger.success("Done.")
            return

        for f in self.folders:
            self._run_one_folder(f)
        logger.success("Done.")
