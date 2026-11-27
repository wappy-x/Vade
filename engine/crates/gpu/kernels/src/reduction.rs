//! Fixed feature-index order shared by the host and the device.
pub const STRIPS:usize=32;
pub const STRIP_COLS:usize=8;

/// Column j belongs to strip (j / 8) % 32. Each strip starts at positive
/// zero and adds rounded terms in ascending column order. Join distances are
/// 16, 8, 4, 2, 1. Only lanes below each distance add their upper partner.
/// Products must be rounded before this function adds them. No fused operation.
#[inline(always)]
pub fn strip_fold(cols:usize,mut term:impl FnMut(usize)->f32)->f32 {
    let mut sum=StripFold::new();
    for column in 0..cols {sum+=term(column);}
    sum.finish()
}

#[derive(Clone,Copy)]
pub struct StripFold {sums:[f32;STRIPS],column:usize}
impl Default for StripFold {
    #[inline(always)]
    fn default()->Self {Self::new()}
}
impl StripFold {
    #[inline(always)]
    pub const fn new()->Self {Self {sums:[0.0;STRIPS],column:0}}
    #[inline(always)]
    pub fn finish(mut self)->f32 {
        macro_rules! join {($distance:literal; $($lane:literal),*)=>{$(self.sums[$lane]+=self.sums[$lane+$distance];)*};}
        join!(16;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        join!(8;0,1,2,3,4,5,6,7);
        join!(4;0,1,2,3);
        join!(2;0,1);
        join!(1;0);
        self.sums[0]
    }
}
impl core::ops::AddAssign<f32> for StripFold {
    #[inline(always)]
    fn add_assign(&mut self,value:f32) {
        macro_rules! add {($($lane:literal),*)=>{match self.column/STRIP_COLS%STRIPS { $($lane=>self.sums[$lane]+=value,)* _=>unreachable!() }};}
        add!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);
        self.column+=1;
    }
}

#[cfg(any(test,target_arch="nvptx64"))]
#[cfg_attr(test,allow(unused_macros))]
macro_rules! warp_fold_join {($sum:ident,$shuffle:expr,$add:expr)=>{{
    let mut distance=(crate::reduction::STRIPS/2) as u32;
    while distance!=0 {
        let other=($shuffle)($sum,distance);
        $sum=($add)($sum,other);
        distance/=2;
    }
}};}
#[cfg(any(test,target_arch="nvptx64"))]
#[cfg_attr(test,allow(unused_imports))]
pub(crate) use warp_fold_join;

/// Expand inside a checked full warp. Only lane zero owns the returned sum.
#[cfg(target_arch="nvptx64")]
macro_rules! warp_fold {($cols:expr,$lane:expr,|$column:ident|$term:expr)=>{{
    let columns=$cols;let lane=$lane;let mut sum=0.0f32;
    let mut first=lane*crate::reduction::STRIP_COLS;
    while first<columns {
        for offset in 0..crate::reduction::STRIP_COLS {
            let $column=first+offset;
            if $column<columns {
                let product:f32=$term;
                core::arch::asm!("add.rn.f32 {sum}, {sum}, {product};",sum=inout(reg32)sum,product=in(reg32)product,options(nomem,nostack));
            }
        }
        first+=crate::reduction::STRIPS*crate::reduction::STRIP_COLS;
    }
    crate::reduction::warp_fold_join!(sum,|value:f32,distance:u32| {
        let other:f32;
        core::arch::asm!("shfl.sync.down.b32 {other}, {sum}, {distance}, 31, -1;",other=out(reg32)other,sum=in(reg32)value,distance=in(reg32)distance,options(nomem,nostack));
        other
    },|mut value:f32,other:f32| {
        core::arch::asm!("add.rn.f32 {sum}, {sum}, {other};",sum=inout(reg32)value,other=in(reg32)other,options(nomem,nostack));
        value
    });
    sum
}};}
#[cfg(target_arch="nvptx64")]
pub(crate) use warp_fold;

/// Butterfly fold of one value per lane (lane = strip). Every lane returns the root.
/// Every lane of the warp must call it together (full-mask shuffles).
#[cfg(target_arch="nvptx64")]
#[inline(always)]
pub(crate) fn warp_root(mut sum:f32)->f32 {
    // SAFETY: Register-only full-mask butterfly shuffles; every caller reaches this with all 32 lanes of its warp
    // (warp-uniform control flow), so each source lane participates. No memory is read or written.
    unsafe {
        macro_rules! step {($d:literal)=>{{
            let other:f32;
            core::arch::asm!("shfl.sync.bfly.b32 {o}, {s}, {d}, 31, -1;",o=out(reg32)other,s=in(reg32)sum,d=const $d,options(nomem,nostack));
            core::arch::asm!("add.rn.f32 {sum}, {sum}, {other};",sum=inout(reg32)sum,other=in(reg32)other,options(nomem,nostack));
        }};}
        step!(16);step!(8);step!(4);step!(2);step!(1);
    }
    sum
}
