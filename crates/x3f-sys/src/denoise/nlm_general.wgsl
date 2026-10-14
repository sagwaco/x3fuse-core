// Configurable NLM windows. The tuned 3x3 pipeline remains the default.
struct Params {
    input_width: u32, input_height: u32, width: u32, height: u32,
    radius: u32, lut_len: u32, template_radius: u32, bin_shift: u32,
}
@group(0) @binding(0) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
fn pixel(x: u32, y: u32) -> vec3<i32> {
    let p = source[y * params.input_width + x];
    return vec3<i32>(i32(p.x & 65535u), i32(p.x >> 16u), i32(p.y));
}
@compute @workgroup_size(8,8)
fn nlm(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let tr = params.template_radius - 1u;
    let halo = params.radius + tr;
    let center = vec3<f32>(pixel(id.x + halo, id.y + halo));
    var estimate = vec3<f32>(0.0);
    var total = vec3<f32>(0.0);
    for (var sy = 0u; sy <= 2u * params.radius; sy++) {
        for (var sx = 0u; sx <= 2u * params.radius; sx++) {
            var distance = 0u;
            for (var dy = 0u; dy <= 2u * tr; dy++) {
                for (var dx = 0u; dx <= 2u * tr; dx++) {
                    let delta = abs(pixel(id.x + params.radius + dx, id.y + params.radius + dy)
                        - pixel(id.x + sx + dx, id.y + sy + dy));
                    distance += u32(delta.x + delta.y + delta.z);
                }
            }
            let weight = weights[min(distance >> params.bin_shift, params.lut_len - 1u)].xyz;
            estimate += weight * (vec3<f32>(pixel(id.x + sx + tr, id.y + sy + tr)) - center);
            total += weight;
        }
    }
    let value = vec3<u32>(clamp(floor(center + estimate / total + 0.5), vec3<f32>(0.0), vec3<f32>(65535.0)));
    destination[id.y * params.width + id.x] = vec2<u32>(value.x | (value.y << 16u), value.z);
}
