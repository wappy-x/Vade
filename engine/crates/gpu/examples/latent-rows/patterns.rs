pub fn name(pattern:u8)->&'static str {
    match pattern {0=>"finite",1=>"NaN_payloads_and_infinities",2=>"denormals_and_signed_zero",_=>"wide_exponents"}
}
pub fn word(pattern:u8,index:usize,query:bool)->f32 {
    let finite=if query {(index as i32%29-14) as f32/64.0} else if index%19==0 {-0.0} else {(index as i32%67-33) as f32/32.0};
    let feature=index%256;let head=if query {index/256} else {index/512%64};let value=!query && index%512>=256;
    match pattern {
        0=>finite,
        1=>match (head%8,query,value,feature%31) {
            (0,true,_,0)|(1,false,false,0)=>f32::from_bits(0x7fc1_2345),
            (2,false,true,0)=>f32::from_bits(0xffc5_4321),
            (3,false,false,1)|(5,true,_,1)=>f32::INFINITY,
            (4,false,true,2)=>f32::NEG_INFINITY,
            (6,false,true,3)=>f32::from_bits(0x7fa1_2345),
            _=>finite,
        },
        2=>if query {if feature%7==0 {-0.0} else {f32::from_bits(207<<23)}} else {
            let bits=match index%8 {0=>0,1=>0x8000_0000,2=>1,3=>0x8000_0001,4=>0x007f_ffff,5=>0x807f_ffff,6=>0x0080_0000,_=>0x8080_0000};f32::from_bits(bits)
        },
        _=>{let exponent=if query {87+(index*17)%81} else {47+(index*29)%161};let sign=if index%3==0 {0x8000_0000} else {0};
            f32::from_bits(sign|((exponent as u32)<<23)|((index as u32).wrapping_mul(73_939)&0x007f_ffff))},
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn special_patterns_cover_payloads_signs_and_exponent_ranges() {
        let nonfinite=(0..32768).map(|index|word(1,index,false)).collect::<Vec<_>>();
        assert!(nonfinite.iter().any(|value|value.to_bits()==0x7fc1_2345));
        assert!(nonfinite.iter().any(|value|value.to_bits()==0xffc5_4321));
        assert!(nonfinite.iter().any(|value|value.to_bits()==0x7fa1_2345));
        assert!(nonfinite.contains(&f32::INFINITY) && nonfinite.contains(&f32::NEG_INFINITY));
        let denorm=(0..32).map(|index|word(2,index,false)).collect::<Vec<_>>();
        assert!(denorm.iter().any(|value|value.to_bits()==1));assert!(denorm.iter().any(|value|value.to_bits()==0x8000_0001));
        assert!(denorm.iter().any(|value|value.to_bits()==0));assert!(denorm.iter().any(|value|value.to_bits()==0x8000_0000));
        assert!(word(2,1,true).is_finite());
        let wide=(0..32768).map(|index|word(3,index,false)).collect::<Vec<_>>();
        assert!(wide.iter().all(|value|value.is_finite()));
        assert!(wide.iter().any(|value|value.abs()>2.0f32.powi(70)) && wide.iter().any(|value|value.abs()<2.0f32.powi(-70)));
        assert!(wide.iter().any(|value|value.is_sign_negative()) && wide.iter().any(|value|value.is_sign_positive()));
        assert_eq!(word(0,19,false).to_bits(),(-0.0f32).to_bits());
        for pattern in 0..=3 {assert!(!name(pattern).is_empty());}
    }
}
