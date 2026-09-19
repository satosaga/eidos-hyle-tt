"""
eidos.lib.power_profile_canvas.data_mixin -- Data loading and axis-limit calculation.

_PowerProfileDataMixin is one of three mixins combined into
PowerProfileCanvas (see canvas.py); it is not a usable class on its own
-- it assumes the final class also provides QWidget (self.update(), etc.)
and the other two mixins' methods/attributes where it calls them.
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from PySide6.QtCore import Slot

import eidos.lib.strategy_selector as rs
from eidos.lib.power_profile_canvas.plot_limits import (
    PROFILE_KEYS,
    PlotLimits,
)

# See draw_mixin.py's comment above its own _DrawMixinBase (and
# _canvas_selftype.py's docstring): same TYPE_CHECKING-only trick, pointed
# at the cycle-free stand-in class rather than the real PowerProfileCanvas,
# so mypy knows self also has QWidget/other-mixin attributes
# (self.request_refresh, self.update(), ...) without any real inheritance
# or circular import at runtime.
if TYPE_CHECKING:
    from eidos.lib.power_profile_canvas._canvas_selftype import _CanvasSelfType
    _DataMixinBase = _CanvasSelfType
else:
    _DataMixinBase = object


class _PowerProfileDataMixin(_DataMixinBase):
    """Data loading, model syncing, and axis-limit calculation.

    Not a usable class on its own -- see this module's docstring.
    """

    @Slot()
    def refresh_from_model(self):
        """Read current selection state from the model and synchronise plots."""
        # 1. Extract only the records to pin as background (excluding Designer temp runs)
        selected_records = [
            r for r in getattr(self.model, "_records", []) 
            if r.get('is_selected', False) and r.get('display_name') != "DESIGNER_TEMP_RUN"
        ]
        # 2. Update the background (cache)
        self.set_target_records(selected_records)
        self._cache_dirty = True 
        self.update()

    def set_target_records(self, records: List[Dict[str, Any]]):
        """Finalise records to render, rebuild the data cache, and normalise values."""
        self._rebuild_data_cache(records)
        
        # Compute limits only when at least one valid record exists (sparse-data resilience)
        if self.data_cache:
            self.calculate_plot_limits()
        
        self.request_refresh()

    def set_axis_mode(self, is_time_mode: bool):
        """Switch axis mode and trigger a redraw (called from outside)."""
        self.is_time_mode = is_time_mode
        self.calculate_plot_limits()
        self.request_refresh()

    def set_cursor_position(self, dist_m):
        """Update the cursor distance and trigger a repaint."""
        self.cursor_dist = dist_m
        self.update()  # kick a redraw

    def _get_plot_arrays(self, data, plot_name):
        """Return the X array and Y array(s) required for drawing, based on plot_name."""
        # Select X axis
        if plot_name == 'Course':
            x = data.get('TIME_COURSE', np.array([])) if self.is_time_mode else data.get('COURSE_DIST', np.array([]))
            return x, data['SLOPE'] * 100, data['ALTITUDE']
        
        x = data['TIME'] if self.is_time_mode else data['DISTANCE']
        
        # Select Y axis and apply unit conversion
        if plot_name == 'Power':
            return x, data['POWER'], None
        elif plot_name == 'Velocity':
            return x, data['SPEED'] * 3.6, None
        elif plot_name == 'WPrime':
            return x, data['WPRIME'], None
        elif plot_name == 'Cumulative':
            y = data['DISTANCE'] / 1000.0 if self.is_time_mode else data['TIME'] / 60.0
            return x, y, None
            
        return x, None, None

    def _get_legend_config(self, plot_name):
        """Return legend configuration for the given plot_name (includes Cumulative)."""
        
        # Base config dict
        configs = {
            'Course': (['Grade', 'Altitude'], 'TopLeft'),
            'Power': (['Target Power', 'Actual Power', 'CP'], 'BottomRight'),
            'Velocity': (['Velocity'], 'BottomRight'),
            'WPrime': (["W' Balance", "W' Max"], 'TopRight'),
            # Cumulative added; 'Cumulative' is a key the drawing engine understands.
            'Cumulative': (['Cumulative'], 'TopLeft') 
        }
        
        return configs.get(plot_name, ([], 'TopRight'))

    def _get_sorted_plot_data(self):
        """Return sorted data pairs based on draw order (ascending TotalTime_s, i.e. fastest first)."""
        if not self.data_cache:
            return [], []

        combined = list(zip(self.data_cache, self.records))
        combined.sort(key=lambda item: item[0]['TotalTime_s'])
        
        # Unpack and return
        sorted_data, sorted_recs = zip(*combined)
        return list(sorted_data), list(sorted_recs)

    def _get_segment_boundaries(self):
        """Compute segment boundary X-coordinates across all data and return as a sorted unique list."""
        all_segment_x = set()
        for d in self.data_cache:
            if d['TargetL_m'].size > 0:
                dist_edges = np.concatenate([[0], np.cumsum(d['TargetL_m'])])
                if self.is_time_mode and d['TIME'].size > 0:
                    # Project distance-based boundaries onto the time axis
                    time_edges = np.interp(dist_edges, d['DISTANCE'], d['TIME'])
                    all_segment_x.update(time_edges)
                else:
                    all_segment_x.update(dist_edges)
        return sorted(list(all_segment_x))

    def calculate_plot_limits(self, temporary_record=None):
        """
        Recompute axis limits for all five subplots from the current data cache.

        Args:
            temporary_record: optional extra data dict appended for limit calculation
                              without modifying the permanent cache (used by Designer preview).

        Returns:
            True if any limit changed, False otherwise.
        """
        all_data_cache = list(self.data_cache)
        if temporary_record is not None:
            all_data_cache.append(temporary_record)

        if not all_data_cache:
            return False

        x_key = 'TIME' if getattr(self, 'is_time_mode', False) else 'DISTANCE'
        
        # --- 1. X-axis computation (no concatenate) ---
        # Build a list of np.max values per array; take the overall maximum. No memory copy.
        x_max_list = [np.max(d[x_key]) for d in all_data_cache if x_key in d and d[x_key].size > 0]
        if not x_max_list:
            return False
        X_MAX = max(x_max_list) * 1.05

        changed = False
        def update_limit(obj, attr, new_val):
            """Set obj.attr to new_val and mark changed=True if the value differs."""
            nonlocal changed
            old_val = getattr(obj, attr)
            if abs(old_val - new_val) > 1e-5:
                setattr(obj, attr, new_val)
                changed = True

        for limit in self.limits.values():
            update_limit(limit, 'X_min', 0.0)
            update_limit(limit, 'X_max', X_MAX)

        if 'Cumulative' not in self.limits:
            self.limits['Cumulative'] = PlotLimits()
            changed = True
        
        update_limit(self.limits['Cumulative'], 'X_max', X_MAX)
        update_limit(self.limits['Cumulative'], 'Y1_min', 0.0)

        # --- 2. Per-metric computation (all switched to individual aggregation) ---

        # Cumulative Y-axis
        if self.is_time_mode:
            all_dist_max = [np.max(d['DISTANCE']) for d in all_data_cache if d['DISTANCE'].size > 0]
            max_km = max(all_dist_max) / 1000.0 if all_dist_max else 10.0
            step = 2.0 if max_km <= 10 else 5.0
            update_limit(self.limits['Cumulative'], 'Y1_max', math.ceil(max_km / step) * step)
        else:
            all_time_max = [np.max(d['TIME']) for d in all_data_cache if d['TIME'].size > 0]
            max_min = max(all_time_max) / 60.0 if all_time_max else 60.0
            step = 5.0 if max_min <= 30 else 10.0
            update_limit(self.limits['Cumulative'], 'Y1_max', math.ceil(max_min / step) * step)

        # Course (Grade / Altitude)
        slopes_min = [np.min(d['SLOPE']) * 100 for d in all_data_cache if d['SLOPE'].size > 0] or [0.0]
        slopes_max = [np.max(d['SLOPE']) * 100 for d in all_data_cache if d['SLOPE'].size > 0] or [0.0]
        s_min, s_max = min(slopes_min), max(slopes_max)
        y1_min = min(0.0, math.floor(s_min / 10) * 10)
        y1_max = max(0.0, math.ceil(s_max / 10) * 10)
        if y1_max - y1_min < 20:
            mid = (y1_min + y1_max) / 2
            y1_min, y1_max = mid - 10, mid + 10
        update_limit(self.limits['Course'], 'Y1_min', y1_min)
        update_limit(self.limits['Course'], 'Y1_max', y1_max)

        alts_min = [np.min(d['ALTITUDE']) for d in all_data_cache if d['ALTITUDE'].size > 0] or [0.0]
        alts_max = [np.max(d['ALTITUDE']) for d in all_data_cache if d['ALTITUDE'].size > 0] or [0.0]
        a_min, a_max = min(alts_min), max(alts_max)
        y2_min, y2_max = math.floor(a_min / 50) * 50, math.ceil(a_max / 50) * 50
        if y2_max - y2_min < 100:
            mid = (y2_min + y2_max) / 2
            y2_min, y2_max = math.floor((mid - 50)/50)*50, math.ceil((mid + 50)/50)*50
        update_limit(self.limits['Course'], 'Y2_min', y2_min)
        update_limit(self.limits['Course'], 'Y2_max', y2_max)
        
        # Power (W)
        p_mins, p_maxs = [0.0], [1.0]
        for d in all_data_cache:
            if d['POWER'].size > 0:
                p_mins.append(np.min(d['POWER'])); p_maxs.append(np.max(d['POWER']))
            if d['TargetP_W'].size > 0:
                p_mins.append(np.min(d['TargetP_W'])); p_maxs.append(np.max(d['TargetP_W']))
            p_mins.append(d['CP_REF']); p_maxs.append(d['CP_REF'])
        p_min, p_max = max(0.0, min(p_mins)), max(p_maxs)
        y1_min, y1_max = math.floor(p_min / 100) * 100, math.ceil(p_max / 100) * 100
        if y1_max - y1_min < 200:
            mid = (y1_min + y1_max) / 2
            y1_min, y1_max = math.floor((mid - 100)/100)*100, math.ceil((mid + 100)/100)*100
        update_limit(self.limits['Power'], 'Y1_min', y1_min)
        update_limit(self.limits['Power'], 'Y1_max', y1_max)

        # Velocity (km/h)
        v_mins = [np.min(d['SPEED']) * 3.6 for d in all_data_cache if d['SPEED'].size > 0] or [0.0]
        v_maxs = [np.max(d['SPEED']) * 3.6 for d in all_data_cache if d['SPEED'].size > 0] or [1.0]
        s_min, s_max = max(0.0, min(v_mins)), max(v_maxs)
        y1_min, y1_max = math.floor(s_min / 10) * 10, math.ceil(s_max / 10) * 10
        if y1_max - y1_min < 20:
            mid = (y1_min + y1_max) / 2
            y1_min, y1_max = math.floor((mid - 10)/10)*10, math.ceil((mid + 10)/10)*10
        update_limit(self.limits['Velocity'], 'Y1_min', y1_min)
        update_limit(self.limits['Velocity'], 'Y1_max', y1_max)

        # W' Balance (J)
        w_mins = [np.min(d['WPRIME']) for d in all_data_cache if d['WPRIME'].size > 0] or [0.0]
        w_maxs = [np.max(d['WPRIME']) for d in all_data_cache if d['WPRIME'].size > 0] or [1.0]
        for d in all_data_cache:
            w_mins.append(0.0); w_maxs.append(d['W_PRIME_MAX'])
        w_min, w_max = max(0.0, min(w_mins)), max(w_maxs)
        y1_min, y1_max = math.floor(w_min / 5000) * 5000, math.ceil(w_max / 5000) * 5000
        if y1_max - y1_min < 10000:
            mid = (y1_min + y1_max) / 2
            y1_min, y1_max = math.floor((mid - 5000)/5000)*5000, math.ceil((mid + 5000)/5000)*5000
        update_limit(self.limits['WPrime'], 'Y1_min', y1_min)
        update_limit(self.limits['WPrime'], 'Y1_max', y1_max)
        
        return changed

    def get_data_at_dist(self, target_dist):
        """
        Identify the Active record index internally without changing the argument,
        then extract and return data at the given distance.
        """
        if not self.data_cache or not self.records:
            return None
        
        # Key step: dynamically identify the Active index
        active_idx = next((i for i, r in enumerate(self.records) if r.get('is_active')), None)
        
        # When showing all records, fall back to [0] if no Active is found;
        # otherwise use the located index.
        idx = active_idx if active_idx is not None else 0
        if idx >= len(self.data_cache): return None
        
        data = self.data_cache[idx]
        
        import bisect

        import numpy as np

        def get_val_at_dist(key_name, target_d, is_mps=False):
            """Look up the value of key_name at target_d via nearest-index lookup (bisect_left against the reference distance array) -- not interpolated."""
            arr = data[key_name]
            ref_dist_key = 'COURSE_DIST' if key_name in ['ALTITUDE', 'SLOPE', 'LAT', 'LON', 'HEADING'] else 'DISTANCE'
            ref_dist_arr = data[ref_dist_key]

            if len(arr) == 0: return 0
            
            i = bisect.bisect_left(ref_dist_arr, target_d)
            i = max(0, min(i, len(arr) - 1))
            
            val = arr[i]
            if key_name == 'HEADING': val = val % 360
            return val * 3.6 if is_mps else val

        # Locate the target power for the current segment
        target_p = 0
        if 'TargetP_W' in data and 'TargetL_m' in data:
            t_l, t_p = data['TargetL_m'], data['TargetP_W']
            cumulative_l = np.cumsum(t_l) 
            t_idx = bisect.bisect_right(cumulative_l, target_dist)
            target_p = t_p[max(0, min(t_idx, len(t_p) - 1))]

        return {
            'dist':    target_dist,
            'time':    get_val_at_dist('TIME', target_dist),  
            'alt':     get_val_at_dist('ALTITUDE', target_dist),
            'grade':   get_val_at_dist('SLOPE', target_dist) * 100,
            'lat':     get_val_at_dist('LAT', target_dist), 
            'lon':     get_val_at_dist('LON', target_dist), 
            'target':  target_p,
            'actual':  get_val_at_dist('POWER', target_dist),
            'cp':      data['CP_REF'],
            'vel':     get_val_at_dist('SPEED', target_dist, is_mps=True),
            'w_bal':   get_val_at_dist('WPRIME', target_dist),
            'w_max':   data['W_PRIME_MAX'],
        }

    def _rebuild_data_cache(self, records: List[Dict[str, Any]]):
        """
        Handle everything from raw data loading to draw-order sorting in one pass,
        building a finalised cache that the drawing engine (step 5) can use immediately.
        """
        self.records = records
        unsorted_items = []

        for record in records:
            # (1) Identify data source
            target_source = record

            # (2) Convert to NumPy arrays based on PROFILE_KEYS
            cache: Dict[str, Any] = {}
            for key, key_string in PROFILE_KEYS.items():
                v = rs.extract_nested_value(target_source, key_string)
                cache[key] = np.array(v, dtype=float) if (v is not None and np.size(v) > 0) else np.array([], dtype=float) 

            # (3) Engineering computations and interpolation (Heading normalisation / TIME_COURSE generation)
            if 'HEADING' in cache and cache['HEADING'].size > 0:
                cache['HEADING'] = np.mod(cache['HEADING'], 360)
            
            out_dist = cache.get('DISTANCE', np.array([]))
            out_time = cache.get('TIME', np.array([]))
            in_dist = cache.get('COURSE_DIST', np.array([]))
            
            if out_dist.size > 0 and out_time.size > 0 and in_dist.size > 0:
                # Back-calculate the time axis corresponding to the distance axis (in_dist)
                cache['TIME_COURSE'] = np.interp(in_dist, out_dist, out_time)
            else:
                cache['TIME_COURSE'] = np.array([], dtype=float)

            # (4) Attach KPIs and attributes. Direct indexing, not
            # extract_nested_value's by-design None-on-missing: every
            # strategy JSON eidos.apps.generator writes always has
            # output.results.{strategy.target_length_list,
            # strategy.target_power_list, kpis.total_time_s} and
            # input.settings.physiological.{cp,w_prime} (the latter
            # required on every SIMULATOR_REGISTRY entry's
            # PhysiologicalSettings -- see core.schema.
            # PhysiologicalSettingsBase), so a real record missing any of
            # them is a bug, not a legitimately-absent field.
            strategy_out = target_source['output']
            cache['TargetL_m']   = np.array(strategy_out['results']['strategy']['target_length_list'], dtype=float)
            cache['TargetP_W']   = np.array(strategy_out['results']['strategy']['target_power_list'], dtype=float)
            cache['TotalTime_s'] = strategy_out['results']['kpis']['total_time_s']
            cache['CP_REF']      = target_source['input']['settings']['physiological']['cp']
            cache['W_PRIME_MAX'] = target_source['input']['settings']['physiological']['w_prime']
            cache['COLOR_HEX']   = record.get('color_hex')

            unsorted_items.append((cache, record))

        # (5) Finalise draw order (sort ascending by TotalTime_s — fastest first)
        # This eliminates repeated sort computation inside paintEvent.
        unsorted_items.sort(key=lambda item: item[0]['TotalTime_s'])

        # (6) Unpack and finalise
        if unsorted_items:
            self.data_cache, self.records = zip(*unsorted_items)
        else:
            self.data_cache, self.records = (), ()
            
        self._cache_dirty = True

    # ------------------------------------------------
    # 3. Geometry & Mapping
    # ------------------------------------------------
