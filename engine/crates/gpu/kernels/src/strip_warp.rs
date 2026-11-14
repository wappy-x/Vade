//! Register helpers for the mHC strip sums: vector loads, separate `mul.rn.f32` and `add.rn.f32` (no fused
//! multiply-add), and straight-line unrolling. The projection strip engines of order-tree-fast are not on this
//! branch (mHC-only order change).

macro_rules! ld4 {($p:expr)=>{{use core::arch::asm;
    let (a,b,c,d):(u32,u32,u32,u32);
    asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$p,options(pure,readonly,nostack));
    [f32::from_bits(a),f32::from_bits(b),f32::from_bits(c),f32::from_bits(d)]
}};}
macro_rules! mul {($a:expr,$b:expr)=>{{use core::arch::asm;let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(pure,nomem,nostack));value}};}
macro_rules! add {($sum:expr,$value:expr)=>{{use core::arch::asm;let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(pure,nomem,nostack));}};}

/// Straight-line copies of `$body` with `$v` = 0 .. $n - 1 (constant indices keep small arrays in registers).
macro_rules! unroll {
    (1,$v:ident,$b:block)=>{{let $v:usize=0;$b}};
    (2,$v:ident,$b:block)=>{{crate::strip_warp::unroll!(1,$v,$b);{let $v:usize=1;$b}}};
    (3,$v:ident,$b:block)=>{{crate::strip_warp::unroll!(2,$v,$b);{let $v:usize=2;$b}}};
    (4,$v:ident,$b:block)=>{{crate::strip_warp::unroll!(2,$v,$b);{let $v:usize=2;$b}{let $v:usize=3;$b}}};
    (5,$v:ident,$b:block)=>{{crate::strip_warp::unroll!(4,$v,$b);{let $v:usize=4;$b}}};
    (8,$v:ident,$b:block)=>{{crate::strip_warp::unroll!(4,$v,$b);{let $v:usize=4;$b}{let $v:usize=5;$b}{let $v:usize=6;$b}{let $v:usize=7;$b}}};
}
pub(crate) use unroll;
pub(crate) use {ld4,mul,add};
