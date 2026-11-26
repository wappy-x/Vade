//! Independent real-checkpoint pins, including the funded latent and index cache append capacity.
//! Historical served receipts stay historical. The expanded-limit selector is
//! absent from this base; its later integration must retain these added bytes.
use engine_model::plan::{ConfigPlan,TextDescription};
use spark_tp::{admission_plan::StateChargePlan,bounded_decoder::{HELD_ROOM,PROMPT_GROUP_MAX,RESERVED_ROWS},row_split::admit_service_memory};

const PLAIN_SPARE:usize=5_767_168;
const PREDICTOR_SPARE:usize=6_291_456;
// Index key and gate caches: 11 latent layers x 2 caches x 2 banks x 128 rows x 128 words x 4 bytes.
const INDEX_SPARE:usize=2_883_584;
const PREDICTOR_INDEX_SPARE:usize=3_145_728;
const E:usize=1_441_792;
const V:usize=2_478_080;
const TAIL_ROW:usize=2_228_224;
const HIDDEN:usize=4096;
const STATE:[(usize,usize);4]=[(4096,296_812_544),(8192,440_991_744),(16384,729_350_144),(32768,1_306_066_944)];
const PREDICTOR_OWN:[(usize,usize);3]=[(8192,26_214_400),(16384,52_428_800),(32768,104_857_600)];
const WORKSPACE:[(usize,usize);3]=[(8192,7_247_444_024),(16384,8_449_128_504),(32768,10_835_720_248)];
const LIMITS:[usize;4]=[1152,2304,4096,8312];
// The independently pinned pre-change table plus 5,767,168 and 2,883,584, once outside max.
const TABLE:[(usize,[usize;4]);3]=[
    (8192,[4_289_810_432,7_611_699_200,12_779_081_728,24_590_241_792]),
    (16384,[4_791_717_888,8_113_606_656,13_280_989_184,25_438_179_328]),
    (32768,[5_945_151_488,9_267_040_256,14_434_422_784,26_591_612_928])];
const RESIDENT:usize=91_776_833_536;
const PREDICTOR_RESIDENT:usize=94_030_004_224;
const BUDGET:usize=109_500_000_000;
const BREAK_EVEN_32768_PREDICTOR:usize=111_054_932_024;
// Actual historical 16K served lines; these were measured before this change.
const HISTORICAL_PLAIN:(usize,usize,usize,usize,usize)=(108_856_568_218,91_810_387_968,8_449_128_504,8_597_051_746,4_783_067_136);
const HISTORICAL_PREDICTOR:(usize,usize,usize,usize,usize)=(109_319_915_520,94_030_004_224,8_449_128_504,6_840_782_792,4_921_479_168);

fn description()->TextDescription {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!("expected text description")};d
}
// Literal state and request oracles. No production charge helper computes them.
fn state(tokens:usize)->usize {152_633_344+44*768*tokens+44*128*(tokens/4)}
fn request(prompt:usize,tokens:usize,limit:usize)->usize {
    let held=E*prompt.min(limit);
    let legacy=2*state(tokens)+2*held+V;
    let early=2*state(prompt.clamp(1,1152))+2*held;let late=2*state(tokens)+held;
    legacy.max(early.max(late)+TAIL_ROW*prompt.clamp(1,256)+V)+PLAIN_SPARE+INDEX_SPARE
}
fn predictor_request(prompt:usize,tokens:usize,limit:usize)->usize {
    let own=PREDICTOR_OWN.iter().find(|p|p.0==tokens).unwrap().1;
    request(prompt,tokens,limit)+2*own+2*(E*(tokens+128).min(limit)-E*prompt.min(limit))+1024*HIDDEN*8+PREDICTOR_SPARE-PLAIN_SPARE+PREDICTOR_INDEX_SPARE-INDEX_SPARE
}
fn workspace(context:usize)->usize {WORKSPACE.iter().find(|p|p.0==context).unwrap().1}

#[test]
fn real_config_state_and_cache_capacity_equal_independent_pins() {
    assert_eq!((HELD_ROOM,PROMPT_GROUP_MAX,RESERVED_ROWS),(128,1024,1152));
    let d=description();let plain=StateChargePlan::new(&d,32768).unwrap();
    assert_eq!(plain.cache_append_reservation_bytes(),PLAIN_SPARE);
    assert_eq!(plain.index_append_reservation_bytes(),INDEX_SPARE);
    assert_eq!(plain.append_reservation_bytes(),8_650_752);
    for (tokens,bytes) in STATE {assert_eq!(state(tokens),bytes);assert_eq!(plain.bytes(tokens).unwrap(),bytes);}
    let mut with_layer=d.with_draft_layer().unwrap().unwrap();with_layer.residual=d.residual;
    let predictor=StateChargePlan::new(&with_layer,32768).unwrap();
    assert_eq!(predictor.cache_append_reservation_bytes(),PREDICTOR_SPARE);
    assert_eq!(predictor.index_append_reservation_bytes(),PREDICTOR_INDEX_SPARE);
    for (tokens,own) in PREDICTOR_OWN {assert_eq!(predictor.bytes(tokens).unwrap()-plain.bytes(tokens).unwrap(),own);}
}

#[test]
fn full_charge_table_adds_spare_capacity_once_and_keeps_historical_receipts() {
    for (context,row) in TABLE {for (i,limit) in LIMITS.into_iter().enumerate() {assert_eq!(request(context,context,limit),row[i]);}}
    let (b,r,w,pool,old)=HISTORICAL_PLAIN;assert_eq!(b-r-w,pool);
    assert_eq!(request(16384,16384,1152),old+PLAIN_SPARE+INDEX_SPARE);
    assert_eq!(request(16384,16384,1152),4_791_717_888);
    let (b,r,w,pool,old)=HISTORICAL_PREDICTOR;assert_eq!(b-r-w,pool);
    assert_eq!(predictor_request(16384,16384,1152),old+PREDICTOR_SPARE+PREDICTOR_INDEX_SPARE);
    assert_eq!(predictor_request(16384,16384,1152),4_930_916_352);
}

#[test]
fn plain_and_predictor_startup_boundaries_include_all_cache_banks() {
    for (context,row) in TABLE {
        let allocated=RESIDENT+workspace(context);let full=row[0];
        assert!(admit_service_memory(allocated,full,BUDGET).is_ok());
        assert_eq!(admit_service_memory(allocated,full,allocated+full),Ok(allocated+full));
        assert!(admit_service_memory(allocated,full,allocated+full-1).is_err());
    }
    let full=predictor_request(32768,32768,1152);assert_eq!(full,6_189_207_552);
    let allocated=PREDICTOR_RESIDENT+workspace(32768);
    assert_eq!(allocated+full,BREAK_EVEN_32768_PREDICTOR);
    assert!(admit_service_memory(allocated,full,BUDGET).is_err());
    assert!(admit_service_memory(allocated,full,BREAK_EVEN_32768_PREDICTOR-1).is_err());
    assert_eq!(admit_service_memory(allocated,full,BREAK_EVEN_32768_PREDICTOR),Ok(BREAK_EVEN_32768_PREDICTOR));
    // The previously served larger negotiated budget still funds this new fee.
    assert!(admit_service_memory(allocated,full,111_885_686_784).is_ok());
}
