// scc_speed_replanner.cpp
#include "scc_replanner/scc_speed_replanner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scc {

std::pair<double, double> SccSpeedReplanner::ProjectToFrenet(
    double px, double py,
    const std::vector<double>& ref_x,
    const std::vector<double>& ref_y,
    const std::vector<double>& cum_s) const
{
    if (ref_x.size() < 2) return {0.0, 0.0};

    int best_k = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < ref_x.size(); ++i) {
        const double dx = ref_x[i] - px;
        const double dy = ref_y[i] - py;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_k = static_cast<int>(i); }
    }
    const double s_val = cum_s[best_k];

    double tx, ty;
    if (best_k + 1 < static_cast<int>(ref_x.size())) {
        tx = ref_x[best_k + 1] - ref_x[best_k];
        ty = ref_y[best_k + 1] - ref_y[best_k];
    } else {
        tx = ref_x[best_k] - ref_x[best_k - 1];
        ty = ref_y[best_k] - ref_y[best_k - 1];
    }
    const double tn = std::hypot(tx, ty);
    if (tn < 1e-6) return {s_val, 0.0};
    tx /= tn; ty /= tn;

    const double rx = px - ref_x[best_k];
    const double ry = py - ref_y[best_k];
    // cross product = lateral offset (좌측 + / 우측 −)
    const double n_val = tx * ry - ty * rx;
    return {s_val, n_val};
}

interface::Trajectory SccSpeedReplanner::Replan(
    const interface::Trajectory& original,
    const interface::PredictObjects& objects,
    const interface::VehicleState& /*ego*/)
{
    interface::Trajectory out = original;
    const size_t N = out.point.size();
    debug_ = SccDebugInfo{};   // reset

    if (N < 2) return out;

    // 1) cum_s + ref xy
    std::vector<double> cum_s(N, 0.0), ref_x(N), ref_y(N), v_orig(N);
    ref_x[0] = original.point[0].x;
    ref_y[0] = original.point[0].y;
    v_orig[0] = original.point[0].speed;
    for (size_t i = 1; i < N; ++i) {
        ref_x[i] = original.point[i].x;
        ref_y[i] = original.point[i].y;
        v_orig[i] = original.point[i].speed;
        const double dx = ref_x[i] - ref_x[i - 1];
        const double dy = ref_y[i] - ref_y[i - 1];
        cum_s[i] = cum_s[i - 1] + std::hypot(dx, dy);
    }
    const double s_max = cum_s.back();
    debug_.cum_s = cum_s;
    debug_.v_original = v_orig;

    // 2) trajectory point 별 예상 도착 시간 (segment 평균 속도로 추정)
    std::vector<double> t_arr(N, 0.0);
    for (size_t i = 1; i < N; ++i) {
        const double v_avg = std::max(cfg_.min_v_for_time,
                                       0.5 * (v_orig[i - 1] + v_orig[i]));
        t_arr[i] = t_arr[i - 1] + (cum_s[i] - cum_s[i - 1]) / v_avg;
    }
    debug_.t_arrival = t_arr;

    // 3) 각 object 의 state[k] 를 frenet 으로 투영, in-lane state 만 keep
    struct LeadCandidate {
        int                  obj_id{0};
        std::vector<double>  t;      // obj_state_dt * k
        std::vector<double>  s;      // 해당 state 의 s (nan = out-of-lane / out-of-range)
    };
    std::vector<LeadCandidate> leads;

    for (const auto& obj : objects.object) {
        LeadCandidate lc;
        lc.obj_id = static_cast<int>(obj.id);
        bool any_in_lane = false;
        for (size_t k = 0; k < obj.state.size(); ++k) {
            const auto& st = obj.state[k];
            const double t_k = static_cast<double>(k) * cfg_.obj_state_dt;
            const auto [s_val, n_val] = ProjectToFrenet(
                st.x, st.y, ref_x, ref_y, cum_s);
            lc.t.push_back(t_k);
            const bool in_lane = std::abs(n_val) <= cfg_.lane_half_width
                              && s_val >= 0.0 && s_val <= s_max;
            lc.s.push_back(in_lane ? s_val : std::nan(""));
            if (in_lane) any_in_lane = true;
        }
        if (any_in_lane) {
            debug_.in_lane_obj_ids.push_back(lc.obj_id);
            leads.push_back(std::move(lc));
        }
    }

    // 4)-5) per trajectory point: t_i 시점 가장 가까운 lead 의 s, gap 계산 → lead-based v_safe
    debug_.closest_lead_s.assign(N, std::nan(""));
    debug_.gap.assign(N, std::nan(""));
    std::vector<double> v_new(N);

    for (size_t i = 0; i < N; ++i) {
        const double t_i = t_arr[i];
        double closest_s = std::nan("");

        for (const auto& lc : leads) {
            if (lc.t.empty()) continue;
            double s_at_t;
            if (t_i <= lc.t.front()) {
                s_at_t = lc.s.front();
            } else if (t_i >= lc.t.back()) {
                s_at_t = lc.s.back();
            } else {
                auto it = std::upper_bound(lc.t.begin(), lc.t.end(), t_i);
                int k1 = static_cast<int>(std::distance(lc.t.begin(), it));
                int k0 = k1 - 1;
                const double s0 = lc.s[k0];
                const double s1 = lc.s[k1];
                if (std::isnan(s0) || std::isnan(s1)) continue;
                const double alpha = (t_i - lc.t[k0]) / (lc.t[k1] - lc.t[k0]);
                s_at_t = s0 + alpha * (s1 - s0);
            }
            if (std::isnan(s_at_t)) continue;
            if (s_at_t > cum_s[i]) {
                if (std::isnan(closest_s) || s_at_t < closest_s) {
                    closest_s = s_at_t;
                }
            }
        }

        double v_i = v_orig[i];
        if (!std::isnan(closest_s)) {
            const double gap = closest_s - cum_s[i];
            const double v_safe = (gap - cfg_.s_min) / cfg_.t_headway;
            v_i = std::min(v_orig[i], std::max(0.0, v_safe));
            debug_.closest_lead_s[i] = closest_s;
            debug_.gap[i] = gap;
        }
        v_new[i] = v_i;
    }
    debug_.v_pre_accel_limit = v_new;

    // 6) Backward pass: 미래 wp 의 낮은 v 에 도달 가능하도록 미리 감속 시작.
    //    v[i] ≤ v[i+1] + max_decel * dt_i  (decel limit, 양수)
    //    예: wp i+1=0 m/s, dt=0.1 → wp i 는 v ≤ 0 + 15*0.1 = 1.5 m/s.
    //        wp i-1 ≤ 1.5 + 1.5 = 3.0 m/s. 점진 감속 램프.
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
        const double dt_i = t_arr[i + 1] - t_arr[i];
        if (dt_i > 0) {
            const double v_dec_limit = v_new[i + 1] + cfg_.max_decel * dt_i;
            if (v_new[i] > v_dec_limit) v_new[i] = v_dec_limit;
        }
    }

    // 7) Forward pass: 가속 limit. v[i] ≤ v[i-1] + max_accel * dt_i.
    //    SCC 거동 위주여서 영향 작지만 brake 후 회복 단계의 jerk 억제.
    for (size_t i = 1; i < N; ++i) {
        const double dt_i = t_arr[i] - t_arr[i - 1];
        if (dt_i > 0) {
            const double v_acc_limit = v_new[i - 1] + cfg_.max_accel * dt_i;
            if (v_new[i] > v_acc_limit) v_new[i] = v_acc_limit;
        }
    }

    // 출력 반영
    for (size_t i = 0; i < N; ++i) {
        out.point[i].speed = static_cast<float>(v_new[i]);
    }
    debug_.v_replanned = v_new;

    return out;
}

}  // namespace scc
