import argparse
import qubic.lib.scanning_strategy.qubic_pointings.qubic_scan_scheduler as qss

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute QUBIC pointing for the Moon."
    )
    # Required arguments
    parser.add_argument("date",        type=str, help="UTC observation date (YYYY-MM-DD)")
    parser.add_argument("source_name", type=str, help="Source name label")
    parser.add_argument("ecsv_file",   type=str, help="Path to the source .ecsv file")
    # Optional arguments with defaults
    parser.add_argument("--az_range",       type=float, default=50.0,   help="Azimuth scan range [deg] (default: 50.0)")
    parser.add_argument("--block_duration", type=float, default=1200.0, help="Duration of each scan block [s] (default: 1200.0)")
    parser.add_argument("--el_start",       type=float, default=40.0,   help="Elevation of the first block [deg] (default: 40.0)")
    parser.add_argument("--el_step",        type=float, default=2.0,    help="Elevation increment between blocks [deg] (default: 2.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    sched = qss.QubicScanScheduler(args.date, args.source_name, az_range=args.az_range)
    sched.load_source_from_file(args.ecsv_file)
    sched.schedule_block_scan(
        block_duration = args.block_duration,
        el_start       = args.el_start,
        el_step        = args.el_step,
    )


if __name__ == "__main__":
    main()
