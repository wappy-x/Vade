//! The made-up checkpoints of the loop driver, from outside the crate. Every name that a check program needs to
//! run the driver on them is public, and `chat` below is written over `Compute`: the same lines run on a device.
use spark_tp::{Result,bounded_decoder::{Compute,Host},loop_decoder::{Driver,fixture::{Fixture,SMALL,WIDE}}};

/// One token as host words: the stream after each visit, the capped logits, the top row.
type Token=(Vec<Vec<f32>>,Vec<f32>,u32);
/// One chat on a backend. The caller of a step owns the work scope of the backend.
fn chat<C:Compute>(compute:&C,fixture:&Fixture,tokens:&[u32],capacity:u64)->Result<Vec<Token>> {
    let driver=Driver::new(compute,fixture,&fixture.description,capacity)?;let mut state=driver.clean()?;
    tokens.iter().map(|&token|compute.work(||{
        let mut streams=Vec::new();
        let step=driver.step_seen(&mut state,token,|_,stream|{streams.push(compute.read(stream)?);Ok(())})?;
        Ok((streams,compute.read(&step.logits)?,step.top))
    })).collect()
}
fn bits(words:&[f32])->Vec<u32> {words.iter().map(|word|word.to_bits()).collect()}
#[test]
fn a_check_program_reaches_the_made_up_checkpoints_and_runs_the_driver_on_them() {
    for config in [SMALL,WIDE] {
        let fixture=Fixture::new(config).unwrap();let d=&fixture.description;let tokens=fixture.chat(5);
        let (first,second)=(chat(&Host,&fixture,&tokens,16).unwrap(),chat(&Host,&fixture,&tokens,16).unwrap());
        assert_eq!(first.len(),tokens.len());
        for ((streams,logits,top),(streams_again,logits_again,top_again)) in first.iter().zip(&second) {
            assert_eq!(streams.len(),d.visits.len());
            assert!(streams.iter().all(|stream|stream.len() as u64==d.hidden && stream.iter().all(|word|word.is_finite())));
            assert_eq!(logits.len() as u64,d.head_rows);assert!(u64::from(*top)<d.head_rows);
            for (stream,again) in streams.iter().zip(streams_again) {assert_eq!(bits(stream),bits(again));}
            assert_eq!((bits(logits),top),(bits(logits_again),top_again));
        }
        // The streams of two visits differ, so the tap hands out each visit and not one value again.
        assert_ne!(bits(&first[0].0[0]),bits(&first[0].0[1]));
    }
    assert!(Fixture::new("{").err().unwrap().contains("the fixture config"));
}
