//! Small nonidentity profile shared by parser and conversion tests.

pub const TABLE: [f32; 24] = [
    0.0, 1.0, 1.0, 12.0, 0.8, 1.1, 0.0, 1.0, 1.0, -8.0, 1.2, 0.9, 0.0, 1.0, 1.0, 16.0, 0.7, 1.3,
    0.0, 1.0, 1.0, -4.0, 1.1, 0.8,
];
pub const TONE: [f32; 6] = [0.0, 0.0, 0.4, 0.6, 1.0, 1.0];

pub fn profile(camera: &str, big_endian: bool) -> Vec<u8> {
    let short = |value: u16| {
        if big_endian {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        }
    };
    let long = |value: u32| {
        if big_endian {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        }
    };
    let words = |values: &[u32]| {
        values
            .iter()
            .flat_map(|&value| long(value))
            .collect::<Vec<_>>()
    };
    let floats = |values: &[f32]| {
        words(
            &values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        )
    };
    let fields = [
        (50708u16, 2u16, format!("{camera}\0").into_bytes()),
        (50936, 2, b"Test look\0".to_vec()),
        (50940, 11, floats(&TONE)),
        (50941, 4, words(&[0])),
        (50981, 4, words(&[2, 2, 2])),
        (50982, 11, floats(&TABLE)),
        (51108, 4, words(&[1])),
        (
            50721,
            10,
            words(&[123, 100, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]),
        ),
    ];
    let mut bytes = Vec::from(if big_endian { b"MM" } else { b"II" });
    bytes.extend(short(0x4352));
    bytes.extend(long(8));
    bytes.extend(short(fields.len() as u16));
    bytes.resize(10 + fields.len() * 12, 0);
    for (index, (tag, kind, data)) in fields.iter().enumerate() {
        let entry = 10 + index * 12;
        bytes[entry..entry + 2].copy_from_slice(&short(*tag));
        bytes[entry + 2..entry + 4].copy_from_slice(&short(*kind));
        let size = match kind {
            2 => 1,
            10 => 8,
            _ => 4,
        };
        bytes[entry + 4..entry + 8].copy_from_slice(&long((data.len() / size) as u32));
        if data.len() <= 4 {
            bytes[entry + 8..entry + 8 + data.len()].copy_from_slice(data);
        } else {
            let offset = bytes.len() as u32;
            bytes[entry + 8..entry + 12].copy_from_slice(&long(offset));
            bytes.extend(data);
        }
    }
    bytes
}
