// Same L1 3x3 patch distances and quantized weight table as the CPU NLM.
// Packed u16 triples occupy two u32s; each invocation owns an entire pixel.
struct Params {
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
    radius: u32,
    lut_len: u32,
    padding0: u32,
    padding1: u32,
}
@group(0) @binding(0) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

// 16x16 output pixels plus the largest supported halo (search radius 10 + 1).
var<workgroup> pixels: array<vec2<u32>, 1444>;

fn unpack(p: vec2<u32>) -> vec3<i32> {
    return vec3<i32>(i32(p.x & 65535u), i32(p.x >> 16u), i32(p.y));
}

@compute @workgroup_size(16, 16)
fn nlm(@builtin(workgroup_id) group: vec3<u32>,
       @builtin(local_invocation_id) local: vec3<u32>,
       @builtin(local_invocation_index) index: u32) {
    let halo = params.radius + 1u;
    let side = 16u + 2u * halo;
    for (var i = index; i < side * side; i += 256u) {
        let x = group.x * 16u + i % side;
        let y = group.y * 16u + i / side;
        var pixel = vec2<u32>(0u);
        if (x < params.input_width && y < params.input_height) {
            pixel = source[y * params.input_width + x];
        }
        pixels[i] = pixel;
    }
    workgroupBarrier();
    let x = group.x * 16u + local.x;
    let y = group.y * 16u + local.y;
    if (x >= params.width || y >= params.height) { return; }
    let center_index = (local.y + halo) * side + local.x + halo;
    let center = vec3<f32>(unpack(pixels[center_index]));
    var reference_patch: array<vec3<i32>, 9>;
    for (var dy = 0u; dy < 3u; dy++) {
        for (var dx = 0u; dx < 3u; dx++) {
            reference_patch[dy * 3u + dx] = unpack(pixels[center_index - side - 1u + dy * side + dx]);
        }
    }
    var estimate = vec3<f32>(0.0);
    var total = vec3<f32>(0.0);
    let search = 2u * params.radius + 1u;
    for (var sy = 0u; sy < search; sy++) {
        for (var sx = 0u; sx < search; sx++) {
            let other = (local.y + sy + 1u) * side + local.x + sx + 1u;
            var distance = 0u;
            for (var dy = 0u; dy < 3u; dy++) {
                for (var dx = 0u; dx < 3u; dx++) {
                    let delta = abs(reference_patch[dy * 3u + dx] - unpack(pixels[other - side - 1u + dy * side + dx]));
                    distance += u32(delta.x + delta.y + delta.z);
                }
            }
            let weight = weights[min(distance >> 4u, params.lut_len - 1u)].xyz;
            // Accumulate differences to preserve constant fields exactly and
            // retain precision around the large unsigned chroma bias.
            estimate += weight * (vec3<f32>(unpack(pixels[other])) - center);
            total += weight;
        }
    }
    let value = vec3<u32>(clamp(floor(center + estimate / total + 0.5), vec3<f32>(0.0), vec3<f32>(65535.0)));
    destination[y * params.width + x] = vec2<u32>(value.x | (value.y << 16u), value.z);
}
