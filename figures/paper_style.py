"""Figure style for the CAL letter: the look of hand-built architecture-paper charts (Office palette,
black-outlined bars, boxed legend on top, shaded average group, Arial-like font), sized for IEEE columns."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = 3.5      # IEEE single column width (in)
DCOL = 7.16    # double column width (in)

# Excel-2007 / gnuplot-era palette (no orange)
NAVY, BLUE, RED, GREEN, PURPLE, AQUA = "#1F497D", "#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6"
LBLUE, LGREEN, LPURPLE = "#B9CDE5", "#D7E4BD", "#CCC1DA"
GRAY, DGRAY = "#BFBFBF", "#595959"
SHADE = "#F2F2F2"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#000000",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.color": "#D9D9D9",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#000000",
        "legend.fancybox": False,
        "legend.borderpad": 0.25,
        "legend.handlelength": 1.5,
        "legend.handleheight": 0.7,
        "legend.columnspacing": 0.8,
        "legend.handletextpad": 0.4,
        "patch.linewidth": 0.5,
        "patch.edgecolor": "#000000",
        "hatch.linewidth": 0.4,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def shade_group(ax, x0, x1):
    """gray hatched background behind the average group, as in classic arch figures."""
    lo, hi = ax.get_ylim()
    ax.add_patch(Rectangle((x0, lo), x1 - x0, hi - lo, facecolor=SHADE, edgecolor="#BFBFBF", hatch="////",
                           lw=0.0, zorder=0))


def top_legend(ax, ncol, **kw):
    return ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, **kw)
