########################
# pdf_exporter.py
# EIDOS^TT — Strategy Report PDF Generator
#
# Output 1: A4 portrait strategy summary (3 charts + settings table)
# Output 2: A4 portrait stem card sheet (cut-out strips for handlebar stem)
########################

import io

import matplotlib
import numpy as np

matplotlib.use('Agg')
import os
from typing import Any, Dict, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from core.data_manager import extract_gpx_base_name, format_time_mmss

# --------------------------------------------------
# Constants
# --------------------------------------------------
PAGE_W, PAGE_H = A4          # 595.27 x 841.89 pt
MARGIN = 15 * mm

C_ACCENT   = colors.HexColor('#58A6FF')   # accent (blue)

# Per-IF colors
IF_COLORS = {
    1.00: '#F85149',   # red
    0.95: '#D29922',   # amber
    0.90: '#58A6FF',   # blue
    0.70: '#3FB950',   # green
    0.50: '#8B949E',   # gray
}

# --------------------------------------------------
# Data container
# --------------------------------------------------
class StrategyData(NamedTuple):
    """Container for all data required by the PDF generator."""
    course_name: str
    cp: float
    run_set_id: str
    n_seg: int
    seed: int
    # Segment data (shared across all IFs)
    seg_powers: Dict[float, np.ndarray] # {if_val: power_array_W}
    seg_lengths_km: np.ndarray          # segment lengths [km]
    seg_cumulative_km: np.ndarray       # cumulative distances [km]
    finish_times: Dict[float, float]    # {if_val: finish_time_s}
    # Simulation trajectories (for chart rendering)
    trajectories: Dict[float, Any]      # {if_val: SimResult}
    # Course profile
    course_dist_km: np.ndarray          # course distance axis [km]
    course_alt_m: np.ndarray            # altitude [m]
    w_prime: float = 0.0                # initial W' [J] (0 = use traj[0])
    settings: dict = {}                 # physical/physiological/run/engine settings


# --------------------------------------------------
# I. Matplotlib chart generation
# --------------------------------------------------

def _mpl_figure_to_rl_image(fig) -> io.BytesIO:
    """Convert a matplotlib Figure to an in-memory PNG buffer for ReportLab.
    Always rendered at 150 DPI -- no caller has ever varied it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def build_chart_altitude_power(data: StrategyData) -> io.BytesIO:
    """
    Chart 1: altitude profile + power overlay.
    Top panel: altitude [m] vs distance [km].
    Bottom panel: power [W] vs distance [km] (step-style, segment-level).
    """
    fig = plt.figure(figsize=(8, 3.2), facecolor='white')
    gs = GridSpec(2, 1, hspace=0.08, height_ratios=[1.2, 1.8])
    ax_alt = fig.add_subplot(gs[0])
    ax_pwr = fig.add_subplot(gs[1], sharex=ax_alt)

    _apply_light_ax(ax_alt)
    _apply_light_ax(ax_pwr)

    # Altitude profile
    ax_alt.fill_between(data.course_dist_km, data.course_alt_m,
                        alpha=0.25, color='#1A6FA8', linewidth=0)
    ax_alt.plot(data.course_dist_km, data.course_alt_m,
                color='#1A6FA8', linewidth=1.2)
    ax_alt.set_ylabel('Alt (m)', color='#333333', fontsize=8)
    ax_alt.yaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.setp(ax_alt.get_xticklabels(), visible=False)

    # Power (IF=1.00 only)
    powers_100 = data.seg_powers.get(1.00)
    if powers_100 is not None:
        x_step, y_step = _make_step_profile(data.seg_cumulative_km, powers_100)
        ax_pwr.plot(x_step, y_step, color=IF_COLORS.get(1.00, '#F85149'),
                    linewidth=1.8, alpha=0.9)

    ax_pwr.axhline(data.cp, color='#888888', linewidth=0.8,
                   linestyle='--', alpha=0.7, label=f'CP {int(data.cp)}W')
    ax_pwr.set_ylabel('Power (W)', color='#333333', fontsize=8)
    ax_pwr.set_xlabel('Distance (km)', color='#333333', fontsize=8)
    ax_pwr.set_ylim(bottom=0)
    ax_pwr.legend(loc='upper right', fontsize=7, framealpha=0.3,
                  labelcolor='#111111', facecolor='white', edgecolor='#CCCCCC')
    ax_pwr.yaxis.set_major_locator(ticker.MaxNLocator(5))

    fig.suptitle(f'{data.course_name}  —  Altitude & Power Profile',
                 color='#111111', fontsize=10, y=0.98)
    return _mpl_figure_to_rl_image(fig)


def build_chart_velocity(data: StrategyData) -> io.BytesIO:
    """Chart 2: velocity profile (IF=1.00)."""
    fig, ax = plt.subplots(figsize=(8, 2.2), facecolor='white')
    _apply_light_ax(ax)

    traj_100 = data.trajectories.get(1.00)
    if traj_100 is not None:
        ax.plot(traj_100.x_traj / 1000.0, traj_100.v_traj * 3.6,
                color=IF_COLORS.get(1.00, '#F85149'), linewidth=1.8, alpha=0.9)

    ax.set_ylabel('Speed (km/h)', color='#333333', fontsize=8)
    ax.set_xlabel('Distance (km)', color='#333333', fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    fig.suptitle('Velocity Profile', color='#111111', fontsize=10)
    fig.tight_layout()
    return _mpl_figure_to_rl_image(fig)


def build_chart_wprime(data: StrategyData) -> io.BytesIO:
    """Chart 3: W' remaining [%] vs distance (IF=1.00)."""
    fig, ax = plt.subplots(figsize=(8, 2.2), facecolor='white')
    _apply_light_ax(ax)

    traj = data.trajectories.get(1.00)
    w0   = data.w_prime if data.w_prime > 0 else None
    w_traj = getattr(traj, 'w_traj', None) if traj else None

    if traj is not None and w_traj is not None and len(w_traj) > 0:
        ref = w0 if (w0 and w0 > 0) else float(w_traj[0])
        if ref > 0:
            ax.plot(traj.x_traj / 1000.0, w_traj / ref * 100,
                    color=IF_COLORS.get(1.00, '#F85149'), linewidth=1.8, alpha=0.9)
            ax.axhline(0, color='#C0392B', linewidth=0.8, linestyle='--', alpha=0.6)
            ax.fill_between(traj.x_traj / 1000.0, w_traj / ref * 100, 0,
                            where=(w_traj / ref * 100 > 0),
                            alpha=0.08, color=IF_COLORS.get(1.00, '#F85149'))
        else:
            ax.text(0.5, 0.5, "W' data not available",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#666666', fontsize=9)
    else:
        ax.text(0.5, 0.5, "W' data not available",
                transform=ax.transAxes, ha='center', va='center',
                color='#666666', fontsize=9)

    ax.set_ylabel("W' remaining (%)", color='#333333', fontsize=8)
    ax.set_xlabel('Distance (km)', color='#333333', fontsize=8)
    fig.suptitle("W' Balance", color='#111111', fontsize=10)
    fig.tight_layout()
    return _mpl_figure_to_rl_image(fig)


def _apply_light_ax(ax):
    """Apply light theme styling to a matplotlib Axes."""
    ax.set_facecolor('white')
    ax.tick_params(colors='#333333', labelsize=7)
    ax.spines[:].set_color('#CCCCCC')
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _make_step_profile(cumulative_km: np.ndarray, powers: np.ndarray):
    """Generate step-style (x, y) arrays for segment boundary transitions."""
    x = []
    y = []
    prev = 0.0
    for dist, p in zip(cumulative_km, powers):
        x.extend([prev, dist])
        y.extend([p, p])
        prev = dist
    return x, y


# --------------------------------------------------
# II. ReportLab page drawing helpers
# --------------------------------------------------

class PageCanvas:
    """Helper for drawing page elements on a light-background canvas."""

    def __init__(self, c: canvas.Canvas):
        self.c = c

    def fill_background(self):
        """Fill the page background with white."""
        self.c.setFillColor(colors.white)
        self.c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    def header(self, title: str, subtitle: str):
        """Draw the accent bar, title, and subtitle at the top of the page."""
        c = self.c
        # Accent line
        c.setFillColor(C_ACCENT)
        c.rect(MARGIN, PAGE_H - MARGIN - 2*mm, PAGE_W - 2*MARGIN, 0.8*mm, fill=1, stroke=0)
        # Title
        c.setFillColor(colors.black)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(MARGIN, PAGE_H - MARGIN - 10*mm, title)
        c.setFillColor(colors.HexColor('#444444'))
        c.setFont('Helvetica', 9)
        c.drawString(MARGIN, PAGE_H - MARGIN - 15*mm, subtitle)

    def footer(self, text: str):
        """Draw a horizontal rule and footer text at the bottom of the page."""
        c = self.c
        c.setFillColor(colors.HexColor('#CCCCCC'))
        c.rect(MARGIN, MARGIN, PAGE_W - 2*MARGIN, 0.3*mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#666666'))
        c.setFont('Helvetica', 7)
        c.drawString(MARGIN, MARGIN + 2*mm, text)
        c.drawRightString(PAGE_W - MARGIN, MARGIN + 2*mm, 'EIDOS^TT')


# --------------------------------------------------
# III. A4 strategy summary PDF
# --------------------------------------------------

def build_summary_pdf(data: StrategyData, output_path: str):
    """
    Render a one-page A4 PDF:
      Left column (55%): 3 charts (fixed aspect ratio)
      Right column (45%): settings table (single column, 8pt)
    """
    from reportlab.lib.utils import ImageReader
    c = canvas.Canvas(output_path, pagesize=A4)
    pc = PageCanvas(c)

    pc.fill_background()
    course_total_km = data.seg_cumulative_km[-1]
    # Direct indexing, not .get(1.00, 0): IF=1.00 is the nominal strategy and
    # must always be present in finish_times -- a missing entry means the
    # IF=1.00 simulation itself failed upstream, which should surface as a
    # loud error here, not as a printed "00:00" on a card taken into a race.
    best_time = data.finish_times[1.00]
    subtitle = (f"Course: {data.course_name}  |  "
                f"{course_total_km:.2f} km  |  "
                f"CP: {int(data.cp)}W  |  "
                f"IF=1.00 → {format_time_mmss(best_time)}  |  "
                f"{data.run_set_id} N{data.n_seg} S{data.seed}")
    pc.header('EIDOS^TT — Strategy Report', subtitle)
    pc.footer(f"Generated by EIDOS^TT Exporter  |  N{data.n_seg}_S{data.seed}")

    area_top    = PAGE_H - MARGIN - 22*mm
    area_bottom = MARGIN + 8*mm
    area_h      = area_top - area_bottom
    area_w      = PAGE_W - 2 * MARGIN

    col_gap  = 4 * mm
    left_w   = area_w * 0.55   # charts
    right_w  = area_w - left_w - col_gap  # settings
    left_x   = MARGIN
    right_x  = MARGIN + left_w + col_gap

    # Left column: 3 charts (preserve aspect ratio)
    charts = [
        build_chart_altitude_power(data),
        build_chart_velocity(data),
        build_chart_wprime(data),
    ]
    # figsize (8,3.2), (8,2.2), (8,2.2) aspect ratios reflected in PDF dimensions
    aspect_ratios = [3.2/8, 2.2/8, 2.2/8]
    gap = 3 * mm
    y_cursor = area_top
    for buf, aspect in zip(charts, aspect_ratios):
        ch = left_w * aspect
        y_cursor -= ch
        c.drawImage(ImageReader(buf), left_x, y_cursor,
                    width=left_w, height=ch, preserveAspectRatio=False)
        y_cursor -= gap

    # Right column: settings table
    _draw_settings_table(c, data, right_x, area_bottom, right_w, area_h)

    c.showPage()
    c.save()


def _draw_settings_table(c: canvas.Canvas, data: StrategyData,
                          x: float, y_bottom: float, w: float, h: float):
    """Render settings values as a single-column table (key/value rows, 8pt)."""
    s = data.settings
    if not s:
        c.setFillColor(colors.HexColor('#888888'))
        c.setFont('Helvetica', 8)
        c.drawString(x, y_bottom + h/2, 'No settings data')
        return

    SECTION_ORDER = ['physical', 'physiological', 'run', 'engine']
    META_SECTIONS = ('versions', 'git_state')
    extra = [k for k in s if k not in SECTION_ORDER and k not in META_SECTIONS]

    rows = []
    for meta_section in META_SECTIONS:
        meta = s.get(meta_section, {})
        if meta and isinstance(meta, dict):
            rows.append((f'[ {meta_section} ]', None, True))
            for k, v in meta.items():
                rows.append((k, v, False))
    for section in SECTION_ORDER + extra:
        if section not in s:
            continue
        rows.append((f'[ {section} ]', None, True))
        for k, v in s[section].items():
            rows.append((k, v, False))

    font_sz = 8.0
    line_h  = h / len(rows)
    line_h  = max(line_h, 4.0 * mm)

    y = y_bottom + h
    data_row_idx = 0
    for label, val, is_header in rows:
        y -= line_h
        if y < y_bottom:
            break

        if is_header:
            c.setFillColor(colors.HexColor('#DDDDDD'))
            c.rect(x, y, w, line_h * 0.92, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#222222'))
            c.setFont('Helvetica-Bold', font_sz)
            c.drawString(x + 1.5*mm, y + line_h * 0.28, label)
        else:
            if data_row_idx % 2 == 0:
                c.setFillColor(colors.HexColor('#F7F7F7'))
                c.rect(x, y, w, line_h * 0.92, fill=1, stroke=0)
            data_row_idx += 1
            c.setFillColor(colors.HexColor('#444444'))
            c.setFont('Helvetica', font_sz)
            c.drawString(x + 1.5*mm, y + line_h * 0.28, str(label))
            if val is not None:
                val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
                c.setFillColor(colors.black)
                c.setFont('Helvetica-Bold', font_sz)
                c.drawRightString(x + w - 1.5*mm, y + line_h * 0.28, val_str)

        c.setStrokeColor(colors.HexColor('#E0E0E0'))
        c.setLineWidth(0.2)
        c.line(x, y, x + w, y)


# --------------------------------------------------
# IV. Stem card PDF (portrait strips, white background)
#
# Spec:
#   - Card size: 25 mm wide x 92 mm tall (portrait, for handlebar stem)
#   - One card per IF value; up to 5 cards arranged on A4 for printing and cutting
#   - White background (toner-saving); black text
#   - Columns: cumulative distance (km) | target power (W)
#   - Font auto-shrinks to fit all segments on one card
#   - Dashed cut-out border
# --------------------------------------------------

# Card dimensions (portrait)
TZ_W  = 25 * mm   # width
TZ_H  = 92 * mm   # height

# White-background color scheme
TZ_BG        = colors.white
TZ_TEXT      = colors.black
TZ_MUTED     = colors.HexColor('#555555')
TZ_STRIPE    = colors.HexColor('#F0F0F0')
TZ_BORDER    = colors.HexColor('#AAAAAA')
TZ_CUT_LINE  = colors.HexColor('#BBBBBB')

# Per-IF colors (print-friendly on white)
TZ_IF_COLORS = {
    1.00: colors.HexColor('#C0392B'),  # red
    0.95: colors.HexColor('#B7770D'),  # amber
    0.90: colors.HexColor('#1A6FA8'),  # blue
    0.70: colors.HexColor('#1E8449'),  # green
    0.50: colors.HexColor('#666666'),  # gray
}

def _pwr_color_light(power: float, cp: float) -> colors.Color:
    """Return a text color for a power value on a white background."""
    if cp is None or cp <= 0:
        return TZ_TEXT
    r = power / cp
    if r >= 1.05:   return colors.HexColor('#C0392B')
    elif r >= 0.95: return colors.HexColor('#B7770D')
    elif r >= 0.75: return TZ_TEXT
    else:           return colors.HexColor('#1E8449')


def build_tanzaku_pdf(data: StrategyData, output_path: str):
    """
    Generate an A4 PDF with portrait stem cards (25 mm x 92 mm each),
    one per IF value, arranged in two sets (primary + spare) with cut lines.
    White background for toner efficiency.
    """
    c = canvas.Canvas(output_path, pagesize=A4)

    if_vals_order = [1.00, 0.95, 0.90, 0.70, 0.50]
    if_vals_avail = [iv for iv in if_vals_order if iv in data.seg_powers]
    n_seg = len(data.seg_lengths_km)
    n_cards = len(if_vals_avail)

    # Cards arranged side-by-side, centered horizontally
    total_cards_w = n_cards * TZ_W
    start_x = (PAGE_W - total_cards_w) / 2

    # Title area
    title_line1_y = PAGE_H - MARGIN - 6*mm
    title_line2_y = PAGE_H - MARGIN - 12*mm

    # White page
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Title (set 1)
    c.setFillColor(colors.black)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(MARGIN, title_line1_y,
                 f'EIDOS^TT  Stem Card  —  {data.course_name}  |  CP {int(data.cp)}W  |  '
                 f'N{data.n_seg} S{data.seed}')
    c.setFillColor(TZ_MUTED)
    c.setFont('Helvetica', 7)
    c.drawString(MARGIN, title_line2_y,
                 'Cut along dashed lines. Choose one IF card and attach to stem.')

    # Set 1: card bottom Y
    card_y1 = title_line2_y - 6*mm - TZ_H

    # Set 1 cards
    for idx, iv in enumerate(if_vals_avail):
        x = start_x + idx * TZ_W
        _draw_single_tanzaku(c, data, x, card_y1, iv, n_seg)

    # Set 2: 8 mm below set 1
    set2_top_y  = card_y1 - 8*mm
    card_y2     = set2_top_y - TZ_H

    # Title (set 2: spare)
    c.setFillColor(colors.black)
    c.setFont('Helvetica-Bold', 8)
    c.drawString(MARGIN, set2_top_y + 5*mm, '[ spare ]')

    # Set 2 cards
    for idx, iv in enumerate(if_vals_avail):
        x = start_x + idx * TZ_W
        _draw_single_tanzaku(c, data, x, card_y2, iv, n_seg)

    # Footer
    c.setFillColor(TZ_MUTED)
    c.setFont('Helvetica', 6.5)
    c.drawCentredString(PAGE_W / 2, MARGIN,
                        f'EIDOS^TT Exporter  |  N{data.n_seg}_S{data.seed}')

    c.showPage()
    c.save()


def _draw_single_tanzaku(c: canvas.Canvas, data: StrategyData,
                          x: float, y: float,
                          if_val: float, n_seg: int):
    """
    Render one portrait stem card (TZ_W x TZ_H) for a single IF value.
    Columns: cumulative distance (km) | target power (W).
    Font auto-shrinks to fit all segments on one card.
    """
    w, h = TZ_W, TZ_H

    # Background and dashed cut-out border
    c.setFillColor(TZ_BG)
    c.rect(x, y, w, h, fill=1, stroke=0)

    c.setStrokeColor(TZ_CUT_LINE)
    c.setLineWidth(0.5)
    c.setDash([1.5, 1.5])
    c.rect(x, y, w, h, fill=0, stroke=1)
    c.setDash()

    # Header: IF value in color bar
    hdr_h = 11 * mm
    hdr_y = y + h - hdr_h

    if_color = TZ_IF_COLORS.get(if_val, colors.black)
    c.setFillColor(if_color)
    c.rect(x, hdr_y, w, hdr_h, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 9)
    c.drawCentredString(x + w/2, hdr_y + hdr_h*0.55, f'IF {if_val:.2f}')

    # Finish time (direct indexing -- see build_strategy_data_from_export's
    # comment on the equivalent IF=1.00 lookup: a missing if_val here means
    # that intensity's simulation failed upstream and should fail loudly,
    # not print a fabricated "00:00" on this panel)
    ft = data.finish_times[if_val]
    c.setFont('Helvetica', 6.5)
    c.drawCentredString(x + w/2, hdr_y + 1.5*mm, format_time_mmss(ft))

    # Column headers
    col_hdr_h = 5 * mm
    col_hdr_y = hdr_y - col_hdr_h

    pad = 1.5 * mm
    usable_w  = w - 2 * pad
    col_km_w  = usable_w * 0.48   # cumulative km
    col_pwr_w = usable_w * 0.52   # power W

    cols = [
        ('km', x + pad,              col_km_w,  False),
        ('W',  x + pad + col_km_w,  col_pwr_w, True),
    ]

    c.setFillColor(colors.HexColor('#E8E8E8'))
    c.rect(x, col_hdr_y, w, col_hdr_h, fill=1, stroke=0)
    c.setStrokeColor(TZ_BORDER)
    c.setLineWidth(0.3)
    c.line(x, col_hdr_y, x + w, col_hdr_y)
    c.line(x, col_hdr_y + col_hdr_h, x + w, col_hdr_y + col_hdr_h)

    c.setFillColor(TZ_MUTED)
    c.setFont('Helvetica-Bold', 5.5)
    for label, cx, cw, _ in cols:
        c.drawCentredString(cx + cw/2, col_hdr_y + col_hdr_h*0.3, label)

    # Segment rows
    data_area_h = col_hdr_y - y - 1.5*mm
    data_top_y  = col_hdr_y

    powers      = data.seg_powers[if_val]
    powers_base = data.seg_powers.get(1.00, powers)
    is_scaled   = abs(if_val - 1.00) > 0.001
    start_km    = np.concatenate([[0.0], data.seg_cumulative_km[:-1]])

    # Variable row height: proportional to segment distance
    seg_dists = data.seg_lengths_km

    font_size = 8.4
    min_row_h = font_size * 1.5 * 0.353 * mm  # minimum pitch to avoid text overlap

    # Step 1: assign minimum pitch to all rows
    row_hs = np.full(n_seg, min_row_h)
    used_h = row_hs.sum()

    if used_h > data_area_h:
        # Shrink font to fit
        font_size = max(3.5, font_size * (data_area_h / used_h))
        min_row_h = font_size * 1.5 * 0.353 * mm
        row_hs    = np.full(n_seg, data_area_h / n_seg)
    else:
        # Step 2: distribute surplus space proportional to segment distance
        surplus   = data_area_h - used_h
        row_hs   += seg_dists / seg_dists.sum() * surplus

    row_tops = data_top_y - np.cumsum(row_hs) + row_hs

    for i in range(n_seg):
        row_top    = row_tops[i]
        rh         = row_hs[i]
        stripe_bot = row_top - rh
        s_km   = start_km[i]
        p_val  = powers[i]
        p_base = powers_base[i]

        # Alternating stripe
        if i % 2 == 1:
            c.setFillColor(TZ_STRIPE)
            c.rect(x, stripe_bot, w, rh, fill=1, stroke=0)

        text_y = row_top - font_size * 0.353 * mm * 1.4

        # Starting distance
        c.setFillColor(TZ_TEXT)
        c.setFont('Helvetica', font_size)
        c.drawCentredString(cols[0][1] + cols[0][2]/2, text_y, f'{s_km:.2f}')

        # Power: "238 (250)" for sub-max IF, "250" for IF=1.00
        c.setFillColor(_pwr_color_light(p_val, data.cp))
        if is_scaled:
            pwr_str  = str(int(round(p_val)))
            base_str = f'({int(round(p_base))})'
            km_cx    = cols[1][1] + cols[1][2] / 2
            c.setFont('Helvetica-Bold', font_size)
            bold_w   = c.stringWidth(pwr_str, 'Helvetica-Bold', font_size)
            paren_w  = c.stringWidth(f' {base_str}', 'Helvetica-Bold', font_size)
            draw_x   = km_cx - (bold_w + paren_w) / 2
            c.setFillColor(_pwr_color_light(p_val, data.cp))
            c.drawString(draw_x, text_y, pwr_str)
            c.setFillColor(_pwr_color_light(p_base, data.cp))
            c.drawString(draw_x + bold_w, text_y, f' {base_str}')
        else:
            c.setFont('Helvetica-Bold', font_size)
            c.drawCentredString(cols[1][1] + cols[1][2]/2, text_y, str(int(round(p_val))))

        # Divider line
        c.setStrokeColor(colors.HexColor('#DDDDDD'))
        c.setLineWidth(0.2)
        c.line(x + pad, stripe_bot, x + w - pad, stripe_bot)


# --------------------------------------------------
# V. Public interface
# --------------------------------------------------

def export_strategy_pdf(data: StrategyData, output_dir: str) -> tuple[str, str]:
    """
    Generate both strategy PDFs and return their paths.
    Called by eidos.apps.exporter's execute_strategy_export().

    Returns:
        (summary_pdf_path, tanzaku_pdf_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    base = f"{data.course_name}_CP{int(data.cp)}_N{data.n_seg}_S{data.seed}"
    summary_path = os.path.join(output_dir, f"{base}_strategy.pdf")
    tanzaku_path = os.path.join(output_dir, f"{base}_stemcard.pdf")

    build_summary_pdf(data, summary_path)
    build_tanzaku_pdf(data, tanzaku_path)

    return summary_path, tanzaku_path


def build_strategy_data_from_export(target, course, sim_results: dict,
                                    if_list: list, settings: dict = {}) -> StrategyData:
    """
    Factory that builds a StrategyData from the data structures used by eidos.apps.exporter.

    Args:
        target: ExportTarget
        course: CourseProfile
        sim_results: {if_val: SimResult} collected in execute_strategy_export()
        if_list: [1.00, 0.95, ...]
        settings: Passed straight through to StrategyData.settings (the
            right-column settings table build_summary_pdf renders) --
            defaults to empty if the caller has nothing to show there.
    """
    seg_powers = {iv: target.target_power_list * iv for iv in if_list if iv in sim_results}
    seg_lengths_km = target.target_length_list / 1000.0
    seg_cumulative_km = np.cumsum(seg_lengths_km)
    finish_times = {iv: sim_results[iv].finish_time for iv in if_list if iv in sim_results}

    # Course profile (fine resolution)
    course_dist_km = course.s_p_fine / 1000.0
    course_alt_m = course.altitude

    return StrategyData(
        course_name=extract_gpx_base_name(target.gpx_filename),
        cp=target.cp,
        run_set_id=target.run_set_id,
        n_seg=target.n_seg_used,
        seed=target.seed_used,
        seg_powers=seg_powers,
        seg_lengths_km=seg_lengths_km,
        seg_cumulative_km=seg_cumulative_km,
        finish_times=finish_times,
        trajectories=sim_results,
        course_dist_km=course_dist_km,
        course_alt_m=course_alt_m,
        w_prime=target.w_prime,
        settings=settings,
    )