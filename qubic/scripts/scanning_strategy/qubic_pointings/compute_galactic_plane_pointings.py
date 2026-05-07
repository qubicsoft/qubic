import qubic_scan_scheduler as qss

date = '2026-05-07'

sched = qss.QubicScanScheduler(date, 'galactic_plane_center_right', az_range=40.0)
sched.load_source_from_file('may.ecsv')
sched.schedule_elevation_tracking(delta_el=2.0)
#sched.plot_schedule()
#sched.compute_coverage(plot=True)
