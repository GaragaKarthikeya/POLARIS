"""Models, CPU workloads, trace rotations, and the 405-setting parameter grid (COSM's evaluation set)."""
# model: (Simulator --model value, transformer layers)
MODELS = {
    'bloom':    ('bloom_sd', 16),      # BLOOM-1B1
    'deepseek': ('deepseek_sd', 28),   # DeepSeek-R1-1.5B
    'qwen2':    ('qwen2_sd', 24),      # Qwen2-0.5B
}
MODEL_NAMES = {'bloom': 'BLOOM-1B1', 'deepseek': 'DeepSeek-R1-1.5B', 'qwen2': 'Qwen2-0.5B'}

# tag: (COSM trace file, rotated-trace prefix, rotation step in lines or None = len/7)
# Rotated names keep the prefix COSM uses to choose each trace's warm-up length.
WORKLOADS = {
    '10':   ('10_tencentmeet_silent.txt',    '10_tencent',   2000),
    '22':   ('22_xiaomibrowser_loading.txt', '22_browser',   None),
    '30':   ('30_weibo_front.txt',           '30_weibo',     None),
    '40':   ('40_xiaominote.txt',            '40_note',      None),
    '51':   ('51_bilibili_video.txt',        '51_bili',      None),
    '60':   ('60_music_play.txt',            '60_music',     None),
    'pa0':  ('pa0_ludcmp.txt',               'pa0_ludcmp',   None),
    'pa1':  ('pa1_covariance.txt',           'pa1_covar',    None),
    'pa2':  ('pa2_floyd.txt',                'pa2_floyd',    None),
    'sp70': ('sp70_519_lbm.txt',             'sp70_lbm',     None),
    'sp71': ('sp71_520_omnetpp.txt',         'sp71_omnetpp', None),
    'sp72': ('sp72_511_povray.txt',          'sp72_povray',  None),
}
NAMES = {'10': 'Tencent Meeting', '22': 'Browser', '30': 'X', '40': 'Note', '51': 'Video', '60': 'Music',
         'pa0': 'Ludcmp', 'pa1': 'Covariance', 'pa2': 'Floyd', 'sp70': '519.lbm', 'sp71': '520.omnetpp', 'sp72': '511.povray'}
ROTATIONS = 'abcde'
LEVELS = [[16, 32, 64, 128, 256],   # n_PTL (PIM command length, cycles)
          [8, 16, 32],              # pred_th
          [2, 6, 12],               # bus_th
          [240, 480, 960],          # io_send_interval
          [0, 50, 150]]             # idle_threshold
COSM_DEFAULT = '128:16:12:480:0'


def configs():
    return [f'{a}:{b}:{c}:{d}:{e}' for a in LEVELS[0] for b in LEVELS[1] for c in LEVELS[2] for d in LEVELS[3] for e in LEVELS[4]]


def result_dir(model, tag):
    return f'results_{model}_{tag}'


def kernel_weight(kernel, layers):
    """per-layer kernels run once per transformer layer; the output projection runs once per token"""
    return 1 if kernel.endswith('output') else layers
