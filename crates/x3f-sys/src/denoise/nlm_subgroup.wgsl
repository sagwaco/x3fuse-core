// wgpu 27 exposes subgroup operations through Features::SUBGROUP.

struct Params {
    input_width: u32, input_height: u32, width: u32, height: u32,
    radius: u32, lut_len: u32, padding0: u32, padding1: u32,
}
@group(0) @binding(0) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn pixel(x: i32, y: i32) -> vec3<i32> {
    let p = source[u32(clamp(y, 0, i32(params.input_height)-1)) * params.input_width + u32(clamp(x, 0, i32(params.input_width)-1))];
    return vec3<i32>(i32(p.x & 65535u), i32(p.x >> 16u), i32(p.y));
}
fn l1(a: vec3<i32>, b: vec3<i32>) -> u32 {
    let d = abs(a-b);
    return u32(d.x+d.y+d.z);
}
fn column_distance(x: i32, y: i32, dx: i32, dy: i32) -> u32 {
    return l1(pixel(x,y-1), pixel(x+dx,y+dy-1))
         + l1(pixel(x,y), pixel(x+dx,y+dy))
         + l1(pixel(x,y+1), pixel(x+dx,y+dy+1));
}

// Neighbouring lanes share their vertical distance sums. This reduces the
// 3x3 calculation to three pixel distances per lane. Subgroup width is dynamic.
@compute @workgroup_size(128)
fn nlm(@builtin(global_invocation_id) id: vec3<u32>,
       @builtin(subgroup_invocation_id) lane: u32,
       @builtin(subgroup_size) lanes: u32) {
    let halo = i32(params.radius) + 1;
    let x = i32(id.x) + halo;
    let y = i32(id.y) + halo;
    let above = pixel(x,y-1);
    let middle = pixel(x,y);
    let below = pixel(x,y+1);
    let center = vec3<f32>(middle);
    var estimate = vec3<f32>(0.0);
    var total = vec3<f32>(0.0);
    // Invocation ordering within subgroups is implementation dependent.
    // Recompute rather than sharing if the adjacent lane is not adjacent x.
    let left_x = subgroupShuffleUp(id.x, 1u);
    let right_x = subgroupShuffleDown(id.x, 1u);
    let own_left = lane == 0u || left_x + 1u != id.x;
    let own_right = lane + 1u == lanes || right_x != id.x + 1u;
    let radius = i32(params.radius);
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let other = pixel(x+dx,y+dy);
            let column = l1(above,pixel(x+dx,y+dy-1)) + l1(middle,other) + l1(below,pixel(x+dx,y+dy+1));
            var left = subgroupShuffleUp(column, 1u);
            var right = subgroupShuffleDown(column, 1u);
            if (own_left) { left = column_distance(x-1,y,dx,dy); }
            if (own_right) { right = column_distance(x+1,y,dx,dy); }
            let distance = left + column + right;
            let weight = weights[min(distance >> 4u, params.lut_len - 1u)].xyz;
            estimate += weight * (vec3<f32>(other) - center);
            total += weight;
        }
    }
    if (id.x < params.width && id.y < params.height) {
        let value = vec3<u32>(clamp(floor(center + estimate / total + 0.5), vec3<f32>(0.0), vec3<f32>(65535.0)));
        destination[id.y * params.width + id.x] = vec2<u32>(value.x | (value.y << 16u), value.z);
    }
}
