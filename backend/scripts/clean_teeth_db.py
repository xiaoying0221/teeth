from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "teeth.db"
BACKUP_PATH = BASE_DIR / f"teeth.backup_before_clean_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.db"


def fetch_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def add_column_if_missing(conn: sqlite3.Connection, table: str, column_sql: str, column_name: str) -> None:
    if column_name not in fetch_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


def clean_database(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    backup_path = db_path.with_name(f"{db_path.stem}.backup_before_clean{db_path.suffix}")
    backup_path.write_bytes(db_path.read_bytes())

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = OFF")

        if not table_exists(conn, "corrections"):
            raise RuntimeError("corrections 表不存在，无法清洗")

        # 先补齐新结构字段，避免后续 update/insert 出错
        add_column_if_missing(conn, "corrections", "source_type VARCHAR NOT NULL DEFAULT 'derived'", "source_type")
        add_column_if_missing(conn, "corrections", "parent_correction_id INTEGER", "parent_correction_id")
        add_column_if_missing(conn, "corrections", "version INTEGER NOT NULL DEFAULT 1", "version")
        add_column_if_missing(conn, "corrections", "is_latest INTEGER NOT NULL DEFAULT 1", "is_latest")
        add_column_if_missing(conn, "corrections", "is_deleted INTEGER NOT NULL DEFAULT 0", "is_deleted")
        add_column_if_missing(conn, "corrections", "box_index INTEGER NOT NULL DEFAULT 0", "box_index")

        cols = fetch_columns(conn, "corrections")
        required = {"id", "detection_id", "box_index", "x", "y", "width", "height", "corrected_time"}
        missing = required - cols
        if missing:
            raise RuntimeError(f"corrections 表缺少字段: {sorted(missing)}")

        rows = conn.execute(
            """
            SELECT
                id, detection_id, box_index, x, y, width, height,
                original_area, edited_area, intersection_area, union_area,
                iou, difference_ratio, corrected_time,
                source_type, parent_correction_id, version, is_latest, is_deleted
            FROM corrections
            ORDER BY detection_id, box_index, corrected_time ASC, id ASC
            """
        ).fetchall()

        if not rows:
            conn.commit()
            print("未找到 corrections 记录，无需清洗")
            return

        # 规则说明：
        # 1) 同一个 (detection_id, box_index) 只保留一条 latest=1 的当前版本
        # 2) manual 记录保留，但同组里如果存在更晚的 derived/deleted，则 manual 视作历史版本
        # 3) 去掉明显脏数据：负面积、空 bbox、重复完全相同记录
        grouped: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[(int(row["detection_id"]), int(row["box_index"] or 0))].append(row)

        keep_ids: set[int] = set()
        delete_ids: set[int] = set()
        update_payload: list[tuple[Any, ...]] = []

        for (detection_id, box_index), items in grouped.items():
            # 过滤明显脏的几何数据
            valid_items = []
            for item in items:
                width = float(item["width"] or 0)
                height = float(item["height"] or 0)
                x = float(item["x"] or 0)
                y = float(item["y"] or 0)
                if width <= 0 or height <= 0:
                    delete_ids.add(int(item["id"]))
                    continue
                if x < 0 or y < 0:
                    delete_ids.add(int(item["id"]))
                    continue
                valid_items.append(item)

            if not valid_items:
                continue

            # 去重：几何和删除状态完全相同的，保留时间更晚的那条
            dedup_map: dict[tuple, sqlite3.Row] = {}
            for item in valid_items:
                key = (
                    round(float(item["x"]), 2),
                    round(float(item["y"]), 2),
                    round(float(item["width"]), 2),
                    round(float(item["height"]), 2),
                    int(item["is_deleted"] or 0),
                    (item["source_type"] or "derived").strip().lower(),
                )
                prev = dedup_map.get(key)
                if prev is None:
                    dedup_map[key] = item
                    continue
                prev_time = prev["corrected_time"] or ""
                cur_time = item["corrected_time"] or ""
                if cur_time > prev_time or (cur_time == prev_time and int(item["id"]) > int(prev["id"])):
                    delete_ids.add(int(prev["id"]))
                    dedup_map[key] = item
                else:
                    delete_ids.add(int(item["id"]))

            deduped = sorted(dedup_map.values(), key=lambda r: (r["corrected_time"] or "", int(r["id"])))

            # 只保留每组最新一条为 latest=1，其余设为 0
            for idx, item in enumerate(deduped, start=1):
                item_id = int(item["id"])
                source_type = (item["source_type"] or "derived").strip().lower()
                is_deleted = 1 if int(item["is_deleted"] or 0) else 0
                is_latest = 1 if idx == len(deduped) else 0
                version = idx
                keep_ids.add(item_id)
                update_payload.append((source_type, is_deleted, is_latest, version, item_id))

            # 处理 parent_correction_id：指向同组前一版本
            for idx, item in enumerate(deduped, start=0):
                parent_id = int(deduped[idx - 1]["id"]) if idx > 0 else None
                update_payload.append((parent_id, int(item["id"])))

        # 先统一更新版本字段
        conn.executemany(
            """
            UPDATE corrections
            SET source_type = ?, is_deleted = ?, is_latest = ?, version = ?
            WHERE id = ?
            """,
            [payload for payload in update_payload if len(payload) == 5],
        )
        conn.executemany(
            "UPDATE corrections SET parent_correction_id = ? WHERE id = ?",
            [payload for payload in update_payload if len(payload) == 2],
        )

        if delete_ids:
            conn.executemany("DELETE FROM corrections WHERE id = ?", [(item_id,) for item_id in sorted(delete_ids)])

        # 让每组只保留最大的 version 作为 latest=1
        groups = conn.execute(
            "SELECT detection_id, box_index, MAX(version) AS max_version FROM corrections GROUP BY detection_id, box_index"
        ).fetchall()
        for row in groups:
            conn.execute(
                """
                UPDATE corrections
                SET is_latest = CASE WHEN version = ? THEN 1 ELSE 0 END
                WHERE detection_id = ? AND box_index = ?
                """,
                (int(row["max_version"] or 1), int(row["detection_id"]), int(row["box_index"] or 0)),
            )

        conn.commit()
        print(f"清洗完成。备份已保存到: {backup_path}")
        print(f"处理结果: 保留 {conn.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]} 条 corrections 记录")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    clean_database()
