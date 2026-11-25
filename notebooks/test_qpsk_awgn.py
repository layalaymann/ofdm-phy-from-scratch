import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.mapper import qpsk_map, qpsk_demap, awgn, qpsk_ber_theory


def run_qpsk_experiment():
    np.random.seed(42)  
    n_bits = 200_000    
    bits = np.random.randint(0, 2, n_bits)
    symbols = qpsk_map(bits)

    eb_n0_db_range = np.arange(0, 11, 1)   
    ber_sim = []

    for eb_n0_db in eb_n0_db_range:
        
        snr_db = eb_n0_db + 10 * np.log10(2)
        rx_symbols = awgn(symbols, snr_db)
        rx_bits = qpsk_demap(rx_symbols)
        ber = np.mean(bits != rx_bits)
        ber_sim.append(ber)
        print(f"Eb/N0 = {eb_n0_db:2d} dB | Simulated BER = {ber:.6f}")

    ber_theory = qpsk_ber_theory(eb_n0_db_range)

      
    plt.figure(figsize=(8, 5))
    plt.semilogy(eb_n0_db_range, ber_theory, "k-", linewidth=2, label="Theory: Q(sqrt(2 Eb/N0))")
    plt.semilogy(eb_n0_db_range, ber_sim, "ro--", markersize=6, label="Simulated QPSK")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("QPSK BER over AWGN Channel")
    plt.legend()
    plt.tight_layout()

    out_path = ROOT / "results" / "qpsk_ber.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    run_qpsk_experiment()