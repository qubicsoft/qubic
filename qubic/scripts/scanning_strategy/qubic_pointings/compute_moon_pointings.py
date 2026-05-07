import qubic_scan_scheduler as qss

date = '2026-05-07'

sched = qss.QubicScanScheduler(date, 'moon', az_range=50.0)
sched.load_source_from_file('moon.ecsv')
sched.schedule_block_scan(block_duration=1200, el_start=40.0, el_step=2.0)
