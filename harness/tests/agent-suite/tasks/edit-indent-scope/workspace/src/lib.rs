pub fn outer_limit(scale: u32) -> u32 {
    let base = 2;
    let limit = 10;
    base + limit * scale
}

pub mod nested {
    pub struct Inner;

    impl Inner {
        pub fn burst_limit(scale: u32) -> u32 {
            let base = 2;
            let limit = 10;
            base + limit * scale
        }
    }
}
