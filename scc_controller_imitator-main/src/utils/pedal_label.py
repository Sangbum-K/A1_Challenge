"""학습 라벨용 pedal map 적용: target ax(+속도) → 단일 pedal scalar (-1~1).

규칙 (사용자 spec):
  pedal = +accel_pedal (acc 매핑, 양수)   if brake_pedal == 0
        = -brake_pedal (brk 매핑, 음수)   if brake_pedal  > 0   (둘 다면 brake 우선)

pedal_map.get_pedal_values 는 스칼라 호출이라, dense grid 를 사전계산 후
RegularGridInterpolator 로 벡터화 (학습데이터 대량 변환용).
"""
from __future__ import annotations
import numpy as np
from .pedal_map import PedalMap

PEDAL_MAP_CSV = "data/control/pedal_map_a1_f4.csv"


class PedalLabelizer:
    """배포(closed_loop)와 정확히 동일한 pedal_map.get_pedal_values 사용.
    accel↔brake 경계 불연속 때문에 grid 보간 대신 quantize→unique→scalar 호출→map-back.
    quantize step 만큼의 미세오차만 발생 (기본 v 0.05 m/s, a 0.02 m/s²)."""
    def __init__(self, pedal_map_csv: str = PEDAL_MAP_CSV, qv: float = 0.05, qa: float = 0.02):
        self.pm = PedalMap()
        if not self.pm.load_from_file(pedal_map_csv):
            raise RuntimeError(f"pedal_map load 실패: {pedal_map_csv}")
        self.qv = qv; self.qa = qa

    def _scalar(self, v: float, a: float) -> float:
        # numpy 1.24 __array_function__ dispatch 가 대량 scalar 호출 시 드물게(~1e-6)
        # UnboundLocalError/TypeError 를 비결정적으로 던짐 → 동일 값 retry 로 우회 (값 불변).
        for _ in range(8):
            try:
                pv = self.pm.get_pedal_values(float(v), float(a))
                return -pv.brake_pedal if pv.brake_pedal > 1e-9 else pv.accel_pedal
            except (UnboundLocalError, TypeError):
                continue
        pv = self.pm.get_pedal_values(float(v), float(a))
        return -pv.brake_pedal if pv.brake_pedal > 1e-9 else pv.accel_pedal

    def _interp_vec(self, vel, acc, is_accel: bool):
        """_interpolate_2d 의 벡터화 (배열 전체 bilinear 1회). clamp 후 grid 내부라
        pedal_map 의 >=len edge 분기는 도달 불가 → 동일 결과. scalar 678k 호출(numpy
        __array_function__ dispatch segfault) 회피용."""
        pm = self.pm
        vb = np.asarray(pm.velocity_breakpoints, np.float64)
        ab = np.asarray(pm.acc_pedal_acceleration_breakpoints if is_accel
                        else pm.brk_pedal_acceleration_breakpoints, np.float64)
        vals = np.asarray(pm.accel_values if is_accel else pm.brake_values, np.float64)  # (n_vel,n_acc)
        nv, na = len(vb), len(ab)
        velc = np.clip(vel, vb[0], vb[-1]); accc = np.clip(acc, ab[0], ab[-1])
        vi = np.clip(np.searchsorted(vb, velc), 1, nv - 1)
        ai = np.clip(np.searchsorted(ab, accc), 1, na - 1)
        v1 = vb[vi - 1]; v2 = vb[vi]; a1 = ab[ai - 1]; a2 = ab[ai]
        f11 = vals[vi - 1, ai - 1]; f12 = vals[vi - 1, ai]
        f21 = vals[vi, ai - 1];     f22 = vals[vi, ai]
        t = np.where(np.abs(v2 - v1) < 1e-6, 0.0, (velc - v1) / (v2 - v1))
        u = np.where(np.abs(a2 - a1) < 1e-6, 0.0, (accc - a1) / (a2 - a1))
        interp = (1 - t) * (1 - u) * f11 + t * (1 - u) * f21 + (1 - t) * u * f12 + t * u * f22
        return np.clip(interp, 0.0, 1.0)

    def label(self, velocity, acceleration) -> np.ndarray:
        """(...,) velocity & acceleration → pedal scalar (-1~1). brake 우선.
        벡터화 bilinear (exact, quantize 없음) — _scalar 대량호출 segfault 회피."""
        v = np.asarray(velocity, np.float64); a = np.asarray(acceleration, np.float64)
        acc_p = self._interp_vec(v, a, True)
        brk_p = self._interp_vec(v, a, False)
        out = np.where(brk_p > 1e-9, -brk_p, acc_p)
        return out.reshape(v.shape).astype(np.float32)
