"""CPU workloads used in the letter: COSM trace file, short tag, and the rotation step (lines)."""
WORKLOADS = {
    #  tag  : (COSM trace file,                rotated-trace prefix, rotation step or None = len/7)
    '10':   ('10_tencentmeet_silent.txt',     '10_tencent', 2000),
    '22':   ('22_xiaomibrowser_loading.txt',  '22_browser', None),
    '51':   ('51_bilibili_video.txt',         '51_bili',    None),
    'sp70': ('sp70_519_lbm.txt',              'sp70_lbm',   None),
}
NAMES = {'10': 'Tencent Meeting', '22': 'Browser', '51': 'Video', 'sp70': '519.lbm'}
ROTATIONS = 'abcde'        # rotations used in the letter (make_rotations.py writes a..g)
LEVELS = [[16, 32, 64, 128, 256],   # n_PTL (PIM command length, cycles)
          [8, 16, 32],              # pred_th
          [2, 6, 12],               # bus_th
          [240, 480, 960],          # io_send_interval
          [0, 50, 150]]             # idle_threshold
COSM_DEFAULT = '128:16:12:480:0'


def configs():
    return [f'{a}:{b}:{c}:{d}:{e}' for a in LEVELS[0] for b in LEVELS[1] for c in LEVELS[2] for d in LEVELS[3] for e in LEVELS[4]]
