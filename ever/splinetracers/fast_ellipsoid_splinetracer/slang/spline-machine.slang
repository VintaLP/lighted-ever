// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import safe_math;
#define PRE_MULTI 1000
#define LADDER_P -0.1

struct SplineState : IDifferentiable
{
  float2 distortion_parts;
  float2 cum_sum;
  float3 padded_depth;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};

struct ControlPoint : IDifferentiable
{
  float t;
  float4 drgb;
}

SplineState inverse_update_dual(
    in SplineState new_state,
    in ControlPoint new_ctrl_pt,
    in ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(new_state.t - t, 0.f);

  SplineState state = {};
  state.drgb = new_state.drgb - new_ctrl_pt.drgb;
  state.t = t;
  float4 drgb = state.drgb;

  float area = max(drgb.x * dt, 0.f);
  let rgb_norm = safe_div(float3(drgb.y, drgb.z, drgb.w), drgb.x);

  state.logT = max(new_state.logT - area, 0.f);
  const float weight = clip((1-safe_exp(-area)) * safe_exp(-state.logT), 0.f, 1.f);

  state.C = new_state.C - weight * rgb_norm;
  float alpha = -safe_expm1(-area);

  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  if (drgb.x < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, drgb.x) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.padded_depth[0] = state.padded_depth[0] - safe_exp(-state.logT) * segment_depth;

  // Distortion Loss
  // make sure this is the right t. I believe this is correct.
  let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
  state.cum_sum.x = new_state.cum_sum.x - weight;
  state.cum_sum.y = new_state.cum_sum.y - weight * m;
  state.distortion_parts.x = new_state.distortion_parts.x - 2 * weight * m * state.cum_sum.x;
  state.distortion_parts.y = new_state.distortion_parts.y - 2 * weight * state.cum_sum.y;

  return state;
}


[Differentiable]
SplineState update(
    in SplineState state,
    in ControlPoint ctrl_pt,
    no_diff in float t_min,
    no_diff in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(t - state.t, 0.f);

  SplineState new_state;
  new_state.drgb = state.drgb + ctrl_pt.drgb;
  new_state.t = t;
  float4 drgb = state.drgb;

  let area = max(drgb.x * dt, 0.f);
  new_state.logT = max(area + state.logT, 0.f);
  float alpha = -safe_expm1(-area);
  const float weight = clip(alpha * safe_exp(-state.logT), 0.f, 1.f);

  let rgb_norm = safe_div(float3(drgb.y, drgb.z, drgb.w), drgb.x);
  new_state.C = state.C + weight * rgb_norm;

  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  if (drgb.x < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, drgb.x) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.padded_depth[0] = state.padded_depth[0] + safe_exp(-state.logT) * segment_depth;

  // Distortion Loss
  // make sure this is the right t. I believe this is correct.
  let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
  new_state.distortion_parts.x = state.distortion_parts.x + 2 * weight * m * state.cum_sum.x;
  new_state.distortion_parts.y = state.distortion_parts.y + 2 * weight * state.cum_sum.y;
  new_state.cum_sum.x = state.cum_sum.x + weight;
  new_state.cum_sum.y = state.cum_sum.y + weight * m;

  return new_state;
}

struct SplineOutput: IDifferentiable {
  float3 C;
  float depth;
  float distortion_loss;
};

[Differentiable]
SplineOutput extract_color(in SplineState state) {
  return {
      state.C,
      state.padded_depth[0],
      state.distortion_parts.x - state.distortion_parts.y
  };
}
