# mpc/ — HPIPM SCC LMPC fork (state ax constraint fix)

A1_Rule 의 LMPC 코드 (lateral / longitudinal + scc_replanner + scc_batch_synth) 를
이 디렉토리로 fork. **longitudinal MPC 의 state ax constraint 활성화** 가 핵심 변경.

## 왜 fork 했나

원본 A1_Rule 의 `longitudinal_delayed_kinematic_lmpc.cpp` 에서
- state ax bound (`qp[i].idxbx = {1}; qp[i].lbx/ubx = ±max_a`) 가 **주석처리**
- input jerk bound 는 `params_.max_jerk` 사용 — config 값이 **1000** (m/s³, 비현실)

ROS controller (warm-chain 100Hz) 는 매 cycle 새 state 로 reset → ax 누적 안 됨.
그러나 **synth_batch (cold single 100-iter solve)** 는 ax 가 자유롭게 누적 →
출력 ax 가 ±170 m/s² 까지 폭주 (학습 데이터 오염).

## 적용 fix (v3 fork)

`src/app/control/longitudinal_control/.../longitudinal_delayed_kinematic_lmpc.cpp`:

```cpp
constexpr double kMaxAxAbs = 5.0;        // m/s²
constexpr double kMaxJerkAbs = 10.0;     // m/s³
for (int i=1; i<=step_num_+1; ++i) {
    qp[i].idxbx = {1};
    qp[i].lbx = -kMaxAxAbs;
    qp[i].ubx =  kMaxAxAbs;
}
const double j_lim = std::min(params_.max_jerk, kMaxJerkAbs);
for (int i=0; i<step_num_+1; ++i) {
    qp[i].idxbu = {0};
    qp[i].lbu = -j_lim;
    qp[i].ubu =  j_lim;
}
```

결과: ax ±170 → ±5 m/s² 정확히 제한.

## 빌드

```bash
cd mpc/
source /opt/ros/noetic/setup.bash
export CMAKE_PREFIX_PATH=/home/ailab-12/git/A1_Rule/devel:$CMAKE_PREFIX_PATH
catkin build scc_batch_synth
```

`config/`, `resources/`, `src/lib/`, `src/app/communication/` 은 A1_Rule 로 symlink.

## 실행 (synth_batch)

`scripts/run_synth_batch_shards.sh` 가 자동으로 이 binary 사용 (`$SYNTH_OVERRIDE` override 가능):

```bash
bash scripts/run_synth_batch_shards.sh frame_list.txt /raw_base /out_dir 16
```
