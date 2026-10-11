"""Read Booster ROS bags (rosbag2 sqlite3) without ROS: /low_state and /joint_ctrl as numpy arrays.

The CDR layouts were worked out from recorded bags (little-endian, 4-byte encapsulation header):

    LowState   imu_state: rpy[3] gyro[3] acc[3] (float32)
               motor_state_parallel: uint32 count, then count x MotorState
               motor_state_serial:   uint32 count, then count x MotorState
    MotorState 36 bytes: mode(uint8, padded to 4) q dq ddq tau_est (float32) temperature(uint8, padded)
               lost(uint32) reserve[2](uint32)
    LowCmd     cmd_type(uint8, padded to 4), uint32 count, then count x MotorCmd
    MotorCmd   28 bytes: mode(uint8, padded to 4) q dq tau kp kd weight (float32)

"parallel" is the motor side (the two ankle drive motors appear in the ankle slots), "serial" the joint side that
the controller reads and commands; only the parallel side reports temperature. Bags cut short (no metadata.yaml)
read fine, since only the sqlite database is used.
"""
from __future__ import annotations

import glob
import os
import sqlite3
import struct

import numpy as np

_MOTOR_STATE_BYTES = 36
_MOTOR_CMD_BYTES = 28


def _db_path(bag: str) -> str:
    if os.path.isfile(bag):
        return bag
    files = sorted(glob.glob(os.path.join(bag, "*.db3")))
    if not files:
        raise FileNotFoundError(f"no .db3 in {bag}")
    return files[0]


def _rows(db, topic: str):
    return db.execute(
        "select m.timestamp, m.data from messages m join topics t on m.topic_id = t.id "
        "where t.name = ? order by m.timestamp", (topic,)).fetchall()


def _motor_states(data: bytes, off: int):
    n = struct.unpack_from("<I", data, off)[0]
    off += 4
    raw = np.frombuffer(data, dtype=np.uint8, count=n * _MOTOR_STATE_BYTES, offset=off).reshape(n, _MOTOR_STATE_BYTES)
    f = raw.view("<f4")
    return {
        "q": f[:, 1].copy(), "dq": f[:, 2].copy(), "ddq": f[:, 3].copy(), "tau": f[:, 4].copy(),
        "temperature": raw[:, 20].astype(np.float32), "lost": raw[:, 24:28].copy().view("<u4")[:, 0],
    }, off + n * _MOTOR_STATE_BYTES


def read_low_state(bag: str) -> dict[str, np.ndarray]:
    """Arrays over messages: t (s, robot clock), rpy/gyro/acc (T,3), and per motor (T,N) for serial (`q`, `dq`,
    `ddq`, `tau`) and parallel (`par_q`, ..., `par_temperature`, `par_lost`) sides."""
    db = sqlite3.connect(_db_path(bag))
    rows = _rows(db, "/low_state")
    out: dict[str, list] = {k: [] for k in ("t", "rpy", "gyro", "acc")}
    for ts, data in rows:
        imu = struct.unpack_from("<9f", data, 4)
        par, off = _motor_states(data, 4 + 36)
        ser, _ = _motor_states(data, off)
        out["t"].append(ts * 1e-9)
        out["rpy"].append(imu[0:3])
        out["gyro"].append(imu[3:6])
        out["acc"].append(imu[6:9])
        for k, v in ser.items():
            out.setdefault(k, []).append(v)
        for k, v in par.items():
            out.setdefault("par_" + k, []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def read_joint_ctrl(bag: str) -> dict[str, np.ndarray]:
    """Commands as received by the bag: t, and per motor (T,N) q, dq, tau, kp, kd, weight. Empty if not recorded."""
    db = sqlite3.connect(_db_path(bag))
    rows = _rows(db, "/joint_ctrl")
    t, cmds = [], []
    for ts, data in rows:
        n = struct.unpack_from("<I", data, 8)[0]
        f = np.frombuffer(data, dtype="<f4", count=n * _MOTOR_CMD_BYTES // 4, offset=12).reshape(n, 7)
        t.append(ts * 1e-9)
        cmds.append(f.copy())
    if not cmds:
        return {"t": np.zeros(0)}
    c = np.stack(cmds)
    return {"t": np.asarray(t), "q": c[:, :, 1], "dq": c[:, :, 2], "tau": c[:, :, 3],
            "kp": c[:, :, 4], "kd": c[:, :, 5], "weight": c[:, :, 6]}
