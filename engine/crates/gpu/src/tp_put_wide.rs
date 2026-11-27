//! The wide mailbox put: the shape host and kernel share. A message of at least MIN_WORDS words in one row uses the take's
//! block rule, so several SMs copy it. The one-block put stays for shorter messages, where bytes cost little.
/// 32,768 bytes. In the head trace, 178 of 279 messages a step and 10.28 of 11.10 MB are at or above this.
pub const MIN_WORDS:usize=8192;
pub const THREADS:u32=256;
pub const fn admitted(rows:usize,words:usize)->bool {rows==1 && words>=MIN_WORDS}
/// The same rule as `tp_mailbox_take`: one block per THREADS words, at most `most`.
pub const fn blocks(words:usize,most:u32)->u32 {
    let wanted=words.div_ceil(THREADS as usize);
    if wanted>most as usize {most} else {wanted as u32}
}
/// The unit (word, or four-word group) a thread copies first; it then steps by `stride` until the end.
pub const fn first(block:u32,thread:u32,threads:u32)->usize {block as usize*threads as usize+thread as usize}
pub const fn stride(grid:u32,threads:u32)->usize {grid as usize*threads as usize}
#[cfg(all(test,not(target_arch="nvptx64")))]
mod tests {
    use super::*;
    /// Every payload size in the exchange census, in words.
    const CENSUS:[usize;10]=[1168,2048,2192,8192,10240,12288,12448,16384,18432,77440];
    fn word_mode(src:&[u32],own:usize,words:usize,grid:u32)->Vec<Option<u32>> {
        let mut out=vec![None;words];
        for block in 0..grid {for thread in 0..THREADS {let mut i=first(block,thread,THREADS);
            while i<words {assert!(out[i].is_none(),"word {i} has two writers");out[i]=Some(src[own+i]);i+=stride(grid,THREADS);}}}
        out
    }
    fn quad_mode(src:&[u32],own:usize,words:usize,grid:u32)->Vec<Option<u32>> {
        let mut out=vec![None;words];
        for block in 0..grid {for thread in 0..THREADS {let mut i=first(block,thread,THREADS);
            while i<words/4 {for n in 0..4 {assert!(out[4*i+n].is_none(),"word {} has two writers",4*i+n);out[4*i+n]=Some(src[own+4*i+n]);}i+=stride(grid,THREADS);}}}
        out
    }
    #[test]
    fn moved_words_equal_the_source_at_every_census_size_and_block_limit() {
        for words in CENSUS.into_iter().chain([8193,8196,8195]) {
            let src=(0..words*2).map(|i|(i as u32).wrapping_mul(2654435761)).collect::<Vec<_>>();
            for own in [0,words] {for most in [1,2,3,5,6,7,8,16,96] {let grid=blocks(words,most);assert!(grid>=1 && grid<=most);
                let expected=src[own..own+words].iter().map(|&v|Some(v)).collect::<Vec<_>>();
                assert_eq!(word_mode(&src,own,words,grid),expected,"words={words} own={own} grid={grid}");
                if words%4==0 {assert_eq!(quad_mode(&src,own,words,grid),expected,"quads words={words} own={own} grid={grid}");}
            }}
        }
    }
    #[test]
    fn only_long_single_row_messages_are_wide() {
        assert!(admitted(1,8192) && admitted(1,77440) && !admitted(1,8191) && !admitted(2,8192) && !admitted(0,8192));
        assert_eq!(CENSUS.iter().filter(|&&w|admitted(1,w)).count(),7);
    }
    #[test]
    fn the_block_rule_is_the_takes() {
        for words in [1usize,255,256,257,1168,8192,16384,77440] {for most in [1u32,6,96] {assert_eq!(blocks(words,most) as usize,words.div_ceil(256).clamp(1,most as usize));}}
    }
}
