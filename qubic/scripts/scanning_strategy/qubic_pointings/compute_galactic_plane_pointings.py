import argparse
import qubic.lib.scanning_strategy.qubic_pointings.qubic_scan_scheduler as qss

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute QUBIC pointings for a given source."
    )
    # Required arguments
    parser.add_argument("date",          type=str, help="UTC observation date (YYYY-MM-DD)")
    parser.add_argument("source_name",   type=str, help="Source name label")
    parser.add_argument("ecsv_file",     type=str, help="Path to the source .ecsv file")
    
    # Optional arguments with defaults from QubicScanScheduler
    parser.add_argument("--az_range",  type=float, default=40.0, help="Azimuth scan range [deg] (default: 40.0)")
    parser.add_argument("--delta_el",  type=float, default=2.0,  help="Elevation strip height [deg] (default: 2.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    sched = qss.QubicScanScheduler(args.date, args.source_name, az_range=args.az_range)
    sched.load_source_from_file(args.ecsv_file)
    sched.schedule_elevation_tracking(delta_el=args.delta_el)
    sched.save_sampling(f"samplings_{args.source_name}_{args.date}.h5")
    sched.plot_schedule()
    sched.compute_coverage(plot=True)


if __name__ == "__main__":
    main()
