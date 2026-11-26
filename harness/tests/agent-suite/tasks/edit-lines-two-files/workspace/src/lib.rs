pub mod a;
pub mod b;

pub fn handshake_ok() -> bool {
    a::SCHEMA_A == 4 && b::SCHEMA_B == 4
}
