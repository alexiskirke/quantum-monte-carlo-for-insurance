"""Compute combined slopes from the n=3 + n=8 data we already have."""
import numpy as np
from scipy import stats as sp_stats

SEED = 42

def fit_slope(queries, rmses, n_bootstrap=2000):
    lq = np.log(np.array(queries, dtype=float))
    lr = np.log(np.array(rmses, dtype=float))
    m = len(lq)
    slope, _, r_value, _, _ = sp_stats.linregress(lq, lr)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(m, m, replace=True)
        s, _, _, _, _ = sp_stats.linregress(lq[idx], lr[idx])
        boots.append(s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, lo, hi, r_value**2

# Synthetic n=3 data (from run output)
q3_queries = [500, 1500, 2500, 3500, 4500, 5500, 6500]
q3_qrmse = [1362.6, 387.5, 256.3, 173.8, 139.4, 116.7, 70.6]
q3_crmse = [987.2, 579.2, 421.3, 335.3, 315.2, 241.9, 277.9]

# Synthetic n=8 data
q8_queries = [500, 2500, 4500, 6500]
q8_qrmse = [1383.3, 270.3, 111.5, 98.2]
q8_crmse = [989.9, 470.7, 284.6, 244.3]

# NOAA n=3 data
n3_queries = [500, 1500, 2500, 3500, 4500, 5500, 6500]
n3_qrmse = [10283.0, 2577.5, 2062.5, 1296.8, 748.6, 781.1, 641.9]
n3_crmse = [6657.5, 4011.5, 2888.2, 2200.3, 2171.6, 1652.3, 2031.8]

print("=" * 60)
print("SYNTHETIC SLOPES")
print("=" * 60)

# n=3 only
qs3, qlo3, qhi3, qr3 = fit_slope(q3_queries, q3_qrmse)
cs3, clo3, chi3, cr3 = fit_slope(q3_queries, q3_crmse)
print(f"n=3 only (7 points, range {min(q3_queries)}-{max(q3_queries)}):")
print(f"  Q: {qs3:.3f} [{qlo3:.3f}, {qhi3:.3f}] R²={qr3:.3f}")
print(f"  C: {cs3:.3f} [{clo3:.3f}, {chi3:.3f}] R²={cr3:.3f}")

# n=3 + n=8 combined
all_q_queries = q3_queries + q8_queries
all_q_rmse = q3_qrmse + q8_qrmse
all_c_queries = q3_queries + q8_queries
all_c_rmse = q3_crmse + q8_crmse
qs_a, qlo_a, qhi_a, qr_a = fit_slope(all_q_queries, all_q_rmse)
cs_a, clo_a, chi_a, cr_a = fit_slope(all_c_queries, all_c_rmse)
print(f"\nn=3+n=8 combined (11 points, range {min(all_q_queries)}-{max(all_q_queries)}):")
print(f"  Q: {qs_a:.3f} [{qlo_a:.3f}, {qhi_a:.3f}] R²={qr_a:.3f}")
print(f"  C: {cs_a:.3f} [{clo_a:.3f}, {chi_a:.3f}] R²={cr_a:.3f}")

# n=8 only
qs8, qlo8, qhi8, qr8 = fit_slope(q8_queries, q8_qrmse)
cs8, clo8, chi8, cr8 = fit_slope(q8_queries, q8_crmse)
print(f"\nn=8 only (4 points, range {min(q8_queries)}-{max(q8_queries)}):")
print(f"  Q: {qs8:.3f} [{qlo8:.3f}, {qhi8:.3f}] R²={qr8:.3f}")
print(f"  C: {cs8:.3f} [{clo8:.3f}, {chi8:.3f}] R²={cr8:.3f}")

print(f"\n" + "=" * 60)
print("NOAA SLOPES (n=3 only -- n=8 too expensive)")
print("=" * 60)
qsn, qlon, qhin, qrn = fit_slope(n3_queries, n3_qrmse)
csn, clon, chin, crn = fit_slope(n3_queries, n3_crmse)
print(f"n=3 only (7 points):")
print(f"  Q: {qsn:.3f} [{qlon:.3f}, {qhin:.3f}] R²={qrn:.3f}")
print(f"  C: {csn:.3f} [{clon:.3f}, {chin:.3f}] R²={crn:.3f}")

print("\nReference: O(1/√N) = -0.500, O(1/N) = -1.000")

# Also compute Table row for n=8 Exp 6
print(f"\n" + "=" * 60)
print("EXP 6 TABLE ROW FOR n=8")
print("=" * 60)
print(f"n=8: 256 bins, disc_err=$567, Q_RMSE=$145 (k=9), C_RMSE=$326, "
      f"ratio=2.2x, SP_2Q=15700, Oracle_depth=63238")
