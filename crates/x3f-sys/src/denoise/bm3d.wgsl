// The same fixed BM3D profile as bm3d.rs. Each workgroup processes one
// reference group. Deterministic integer accumulation combines every overlapping
// filtered block; each dispatch owns a bounded output tile.
struct Params {
    input_width: u32, image_width: u32, image_height: u32, width: u32,
    height: u32, x: u32, y: u32, reference_x: u32,
    reference_y: u32, reference_columns: u32, reference_rows: u32, wiener: u32,
    sigma: f32, padding0: u32, padding1: u32, padding2: u32,
}
@group(0) @binding(0) var<storage, read> noisy: array<f32>;
@group(0) @binding(1) var<storage, read> guide: array<f32>;
@group(0) @binding(2) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> tables: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn index(position: vec2<u32>, coefficient: u32) -> u32 {
    let x = position.x + 32u - params.x + coefficient % 8u;
    let y = position.y + 32u - params.y + coefficient / 8u;
    return y * params.input_width + x;
}
// One workgroup owns a reference group. Collaboratively transform all
// coefficients in shared memory instead of spilling large per-thread arrays.
var<workgroup> coefficients: array<f32,1024>;
var<workgroup> scratch: array<f32,1024>;
var<workgroup> distances: array<f32,81>;
var<workgroup> selected: array<vec2<u32>,8>;
var<workgroup> group_count: u32;
var<workgroup> group_weight: f32;
fn transform(lane: u32, inverse: bool, planes: u32) {
    let y = lane / 8u;
    let x = lane % 8u;
    for (var plane = 0u; plane < planes; plane++) {
        for (var block = 0u; block < 8u; block++) {
            let base = plane * 512u + block * 64u;
            var sum = 0.0;
            for (var k = 0u; k < 8u; k++) {
                let weight = select(tables[x*8u+k], tables[k*8u+x], inverse);
                sum += coefficients[base + y*8u+k] * weight;
            }
            scratch[base + lane] = sum;
        }
    }
    workgroupBarrier();
    for (var plane = 0u; plane < planes; plane++) {
        for (var block = 0u; block < 8u; block++) {
            let base = plane * 512u + block * 64u;
            var sum = 0.0;
            for (var k = 0u; k < 8u; k++) {
                let weight = select(tables[y*8u+k], tables[k*8u+y], inverse);
                sum += scratch[base + k*8u+x] * weight;
            }
            coefficients[base + lane] = sum;
        }
    }
    workgroupBarrier();
}
fn haar(values: ptr<function, array<f32,8>>, n: u32, inverse: bool) {
    var size = select(n, 2u, inverse);
    while (size >= 2u && size <= n) {
        var temp = *values;
        let half = size / 2u;
        for (var i = 0u; i < half; i++) {
            if (inverse) {
                temp[2u*i] = ((*values)[i] + (*values)[half+i]) * 0.7071067811865476;
                temp[2u*i+1u] = ((*values)[i] - (*values)[half+i]) * 0.7071067811865476;
            } else {
                temp[i] = ((*values)[2u*i] + (*values)[2u*i+1u]) * 0.7071067811865476;
                temp[half+i] = ((*values)[2u*i] - (*values)[2u*i+1u]) * 0.7071067811865476;
            }
        }
        *values = temp;
        size = select(size / 2u, size * 2u, inverse);
    }
}
// Quantize weights at 2^20, then accumulate weighted samples with 8
// fractional bits in a signed 64-bit sum represented by two atomic u32s.
// Integer addition is order-independent. At most 36 reference groups x 8
// blocks contribute to a pixel; the u32 weight sum is bounded by 288*2^20.
// This avoids both a native atomic64 requirement and floating-point races.
fn accumulate(dest: u32, sample: f32, weight: f32) {
    let w = u32(round(weight * 1048576.0));
    let value = sample * f32(w) * 256.0;
    let magnitude = abs(value);
    var high = u32(floor(magnitude / 4294967296.0));
    var low = u32(magnitude - f32(high) * 4294967296.0);
    if (value < 0.0) {
        high = ~high + select(0u, 1u, low == 0u);
        low = 0u - low;
    }
    let previous = atomicAdd(&sums[dest*3u], low);
    let carry = select(0u, 1u, previous + low < previous);
    atomicAdd(&sums[dest*3u+1u], high + carry);
    atomicAdd(&sums[dest*3u+2u], w);
}
@compute @workgroup_size(64)
fn filter_groups(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if (id.x >= params.reference_columns * params.reference_rows) { return; }
    let reference = vec2<u32>(params.reference_x + 4u*(id.x % params.reference_columns), params.reference_y + 4u*(id.x / params.reference_columns));
    let limit = params.sigma * params.sigma * select(4.0, 2.0, params.wiener != 0u);
    // Candidate distances are independent, so evaluate them across the group.
    for (var i = lane; i < 81u; i += 64u) {
        let candidate = vec2<i32>(reference) + vec2<i32>(i32(i % 9u)*2-8, i32(i / 9u)*2-8);
        var distance = 3.402823e38;
        if (i != 40u && all(candidate >= vec2<i32>(0)) && candidate.x < i32(params.image_width) && candidate.y < i32(params.image_height)) {
            distance = 0.0;
            for (var k = 0u; k < 64u; k++) {
                let delta = guide[index(reference,k)] - guide[index(vec2<u32>(candidate),k)];
                distance += delta * delta;
            }
            distance /= 64.0;
        }
        distances[i] = distance;
    }
    workgroupBarrier();
    if (lane == 0u) {
        var best: array<f32,8>;
        for (var i = 0u; i < 8u; i++) { selected[i] = reference; best[i] = 3.402823e38; }
        best[0] = 0.0;
        var count = 1u;
        for (var i = 0u; i < 81u; i++) {
            let distance = distances[i];
            if (distance > limit || distance >= best[7]) { continue; }
            var insert = 7u;
            while (insert > 1u && distance < best[insert-1u]) {
                best[insert] = best[insert-1u]; selected[insert] = selected[insert-1u]; insert--;
            }
            best[insert] = distance;
            selected[insert] = vec2<u32>(vec2<i32>(reference) + vec2<i32>(i32(i % 9u)*2-8, i32(i / 9u)*2-8));
            count = min(count + 1u, 8u);
        }
        group_count = 1u;
        while (group_count * 2u <= count) { group_count *= 2u; }
    }
    workgroupBarrier();
    let n = group_count;
    for (var i = 0u; i < 8u; i++) {
        coefficients[i*64u+lane] = noisy[index(selected[i],lane)];
        if (params.wiener != 0u) { coefficients[512u+i*64u+lane] = guide[index(selected[i],lane)]; }
    }
    workgroupBarrier();
    transform(lane, false, select(1u, 2u, params.wiener != 0u));
    var values: array<f32,8>;
    for (var i = 0u; i < n; i++) { values[i] = coefficients[i*64u+lane]; }
    haar(&values, n, false);
    var norm = 0.0;
    if (params.wiener != 0u) {
        var estimates: array<f32,8>;
        for (var i = 0u; i < n; i++) { estimates[i] = coefficients[512u+i*64u+lane]; }
        haar(&estimates, n, false);
        for (var i = 0u; i < n; i++) {
            let power = estimates[i] * estimates[i];
            let gain = power / (power + params.sigma * params.sigma);
            values[i] *= gain; norm += gain * gain;
        }
    } else {
        for (var i = 0u; i < n; i++) {
            if (abs(values[i]) < 2.7 * params.sigma) { values[i] = 0.0; } else { norm += 1.0; }
        }
    }
    haar(&values, n, true);
    for (var i = 0u; i < n; i++) { coefficients[i*64u+lane] = values[i]; }
    distances[lane] = norm;
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i++) { total += distances[i]; }
        group_weight = 1.0 / max(total, 1.0);
    }
    workgroupBarrier();
    transform(lane, true, 1u);
    for (var i = 0u; i < n; i++) {
        let p = selected[i] + vec2<u32>(lane % 8u, lane / 8u);
        if (p.x < params.x || p.y < params.y || p.x >= params.x + params.width || p.y >= params.y + params.height) { continue; }
        let dest = (p.y - params.y) * params.width + p.x - params.x;
        let w = group_weight * tables[64u+lane];
        accumulate(dest, coefficients[i*64u+lane], w);
    }
}
@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width * params.height) { return; }
    let weight = f32(atomicLoad(&sums[id.x * 3u + 2u]));
    var value = noisy[(id.x / params.width + 32u) * params.input_width + id.x % params.width + 32u];
    if (weight > 0.0) {
        let low = f32(atomicLoad(&sums[id.x * 3u]));
        let high = f32(bitcast<i32>(atomicLoad(&sums[id.x * 3u+1u])));
        value = (high * 4294967296.0 + low) / (weight * 256.0);
    }
    output[id.x] = value;
}
