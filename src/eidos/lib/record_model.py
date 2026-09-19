#########################
# record_model.py
#########################
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt
from PySide6.QtGui import QColor

from eidos.lib.visual_profile import get_record_style

logger = logging.getLogger(__name__)


class StrategyRecordModel(QAbstractListModel):
    """
    Qt list model that exposes optimization strategy records to the Viewer UI.
    """

    TIME_KEYS = ['output', 'results', 'kpis', 'total_time_s']

    STRATEGY_ENTITY_ROLE = Qt.ItemDataRole.UserRole      # 256 - full record reference (for Designer)
    STRATEGY_ATTR_ROLE   = Qt.ItemDataRole.UserRole + 1  # 257 - lightweight copy (for Delegate)
    ACTIVE_STATE_ROLE    = Qt.ItemDataRole.UserRole + 2  # 258 - radio button state

    # ----------------------------------------
    # Helpers
    # ----------------------------------------
    @staticmethod
    def record_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
        """Stable identity for a record, independent of list order or sort position."""
        return (
            str(record['run_set_id']),
            str(record['N_seg_file']),
            str(record['Seed_file']),
        )

    def capture_state(self) -> Dict[Tuple[str, str, str], Dict[str, bool]]:
        """Snapshot is_selected/is_active per record, keyed by record_key, for restoring across a rebuild."""
        return {
            self.record_key(record): {
                'is_selected': record.get('is_selected', False),
                'is_active': record.get('is_active', False),
            }
            for record in self._records
        }

    # ----------------------------------------
    # Color computation
    # ----------------------------------------
    def load_and_color_records(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Load raw records, compute color based on finish time, and embed color info.
        Raises immediately on malformed record structure.
        """
        if not raw_records:
            return []

        time_data_list = []
        for i, r in enumerate(raw_records):
            try:
                curr: Any = r
                for key in self.TIME_KEYS:
                    curr = curr[key]
                time_val = float(curr)
            except (KeyError, TypeError, ValueError) as e:
                logger.error("CRITICAL ERROR: Failed to extract 'total_time_s' at index %d.", i)
                logger.error("Path tried: %s", self.TIME_KEYS)
                raise e

            time_data_list.append({
                'time': time_val,
                'original_index': i
            })

        # 1. Sort by finish time
        temp_df = pd.DataFrame(time_data_list).sort_values(by='time', ascending=True).reset_index(drop=True)

        # 2. Prepare time list for color computation
        all_times_for_color = temp_df['time'].tolist()
        N_total = len(temp_df)

        # Pre-allocated with None placeholders, but every index is guaranteed
        # to be overwritten below (one row per raw_records entry, and any
        # extraction failure above already raised instead of leaving a gap).
        processed_records: List[Dict[str, Any]] = [None] * len(raw_records)  # type: ignore[list-item]

        # 3. Attach style info to each record
        for _, row in temp_df.iterrows():
            original_index = int(row['original_index'])
            current_time = row['time']

            record = raw_records[original_index]

            color_rgb_adjusted = get_record_style(
                record, current_time, all_times_for_color, N_total
            )

            hex_color = '#%02x%02x%02x' % tuple(int(c*255) for c in color_rgb_adjusted)

            record['color_hex'] = hex_color
            record['color_rgb'] = color_rgb_adjusted
            record['time'] = current_time

            processed_records[original_index] = record

        return processed_records

    # ----------------------------------------
    # Initialization
    # ----------------------------------------
    def __init__(
        self,
        raw_records: List[Dict[str, Any]],
        parent: Optional[QObject] = None,
        previous_state: Optional[Dict[Tuple[str, str, str], Dict[str, bool]]] = None,
    ):
        super().__init__(parent)

        colored_records = self.load_and_color_records(raw_records)

        # Sort: Designer temp run at top, then ascending by finish time
        self._records = sorted(
            colored_records,
            key=lambda r: (
                r['display_name'] == "DESIGNER_TEMP_RUN",  # True(1) sorts last before reversal
                -r['time']                                 # larger time sorts last before reversal
            ),
            reverse=True  # reverse puts Designer first, then fastest time first
        )

        if previous_state is None:
            # First-ever load in this session: fall back to the original default.
            for i, record in enumerate(self._records):
                record['is_selected'] = (i == 0)  # only first record selected initially
                record['is_active'] = (i == 0)    # only first record active initially
        else:
            # Rebuild (e.g. after Create Design adds a record): carry over each
            # record's prior checkbox/radio state instead of resetting to the
            # first row. A record with no prior entry (newly added) starts
            # unselected/inactive so it doesn't steal focus from the user's
            # existing selection.
            for record in self._records:
                state = previous_state.get(self.record_key(record), {})
                record['is_selected'] = state.get('is_selected', False)
                record['is_active'] = state.get('is_active', False)

        # Cache for identifying the latest seed-0 run
        self.latest_seed0_run_id = ""
        self.update_latest_seed0_cache()

    def update_latest_seed0_cache(self):
        """Update the cached run_set_id of the most recent seed-0 strategy."""
        max_id = ""
        for rec in self._records:
            if str(rec['Seed_file']) == '0':
                run_id = str(rec['run_set_id'])
                if run_id > max_id:
                    max_id = run_id
        self.latest_seed0_run_id = max_id

    # ----------------------------------------
    # QAbstractListModel required methods
    # ----------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of records, or 0 for a valid parent index."""
        if parent.isValid():
            return 0
        return len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return 1; the model is a flat list with a single column."""
        return 1

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role.

        Roles:
            STRATEGY_ENTITY_ROLE: full record dict (used by Designer).
            STRATEGY_ATTR_ROLE: lightweight copy with the fields the Delegate draws as columns.
            ACTIVE_STATE_ROLE: bool radio-button state.
            Qt.ItemDataRole.DisplayRole: display_name string.
            Qt.ItemDataRole.DecorationRole: QColor from the record's color_hex.
            Qt.ItemDataRole.CheckStateRole: Qt.CheckState.Checked / Qt.CheckState.Unchecked.
        """
        if not index.isValid() or index.row() >= len(self._records):
            return None

        record = self._records[index.row()]

        if role == self.STRATEGY_ENTITY_ROLE:  # full record reference for Designer
            return record

        if role == self.STRATEGY_ATTR_ROLE:    # lightweight copy for Delegate
            return {
                "Seed_file": str(record["Seed_file"]),
                "N_seg_file": str(record["N_seg_file"]),
                "run_set_id": str(record["run_set_id"]),
                "strategy_set_dir": str(record["strategy_set_dir"]),
                "time": record["time"],
            }

        if role == self.ACTIVE_STATE_ROLE:     # radio button state
            return record.get('is_active', False)

        if role == Qt.ItemDataRole.DisplayRole:
            return record["display_name"]

        if role == Qt.ItemDataRole.DecorationRole:
            return QColor(record.get('color_hex', '#888888'))

        if role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if record.get('is_selected', True) else Qt.CheckState.Unchecked

        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        """Update record state for the given role and emit dataChanged.

        Qt.ItemDataRole.CheckStateRole toggles is_selected independently per record.
        ACTIVE_STATE_ROLE applies exclusive radio-button selection across all records.
        """
        if not index.isValid():
            return False

        record = self._records[index.row()]

        # Checkbox update (independent per record)
        if role == Qt.ItemDataRole.CheckStateRole:
            is_selected = (value == Qt.CheckState.Checked)
            if record.get('is_selected') != is_selected:
                record['is_selected'] = is_selected
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                return True

        # Radio button update (exclusive; allows all-off)
        if role == self.ACTIVE_STATE_ROLE:
            # Treat incoming value as a toggle signal; derive new absolute state by inverting current
            target_new_state = not record.get('is_active', False)

            # Apply new state across all records
            for i, r in enumerate(self._records):
                r['is_active'] = (i == index.row()) and target_new_state

            # Notify the entire model so UI and model flags stay in sync
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._records) - 1, 0),
                [self.ACTIVE_STATE_ROLE]
            )
            return True

        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags; adds Qt.ItemFlag.ItemIsUserCheckable to the base flags."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return QAbstractListModel.flags(self, index) | Qt.ItemFlag.ItemIsUserCheckable

    def set_all_records_selected(self, select_state: bool):
        """Set the is_selected state of all records at once."""
        if not self._records:
            return

        for record in self._records:
            record['is_selected'] = select_state

        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._records) - 1, 0)

        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.CheckStateRole])