use spark_serve::budget::{HybridBudget,HybridRankLedger};
#[test]
fn hybrid_pool_edges_and_owners() {
    use spark_serve::protocol::{DraftMode as D,typed_mtp_admission}; for mode in [D::Auto,D::Plain,D::Strict] { assert!(typed_mtp_admission(Some(1),false,mode).is_ok()); assert!(mode.require_typed_mtp().is_ok()); for active in [None,Some(0),Some(2),Some(32)] { assert!(typed_mtp_admission(active,false,mode).unwrap_err().contains("--max-active 1")); } assert!(typed_mtp_admission(Some(1),true,mode).unwrap_err().contains("--draft-model")); }
    for mode in [D::Fast,D::Sampled] { assert!(typed_mtp_admission(Some(1),false,mode).unwrap_err().contains("byte-identical")); assert!(mode.require_typed_mtp().unwrap_err().contains("seeded sampling")); }
    for (resident,shared,request,cache,retired) in [(100,20,30,11,7),(0,0,1,0,0),(1000,500,600,100,200)] {
        let owned=2*request+cache+retired; let pool=HybridBudget::new(resident+shared+owned,resident,shared,None).unwrap();
        assert_eq!(pool.sessions,owned); assert!(pool.fits(Some(owned))); assert!(!pool.fits(Some(owned+1))); assert!(!pool.fits(None)); assert!(!pool.fits(Some(0)));
        assert_eq!(pool.active_limit(Some(2),request).unwrap(),2); assert!(pool.active_limit(Some(owned/request+1),request).is_err()); assert!(pool.active_limit(None,0).is_err());
    }
    for (total,resident,shared,cap) in [(10,11,0,None),(usize::MAX,usize::MAX,1,None),(10,5,5,None),(10,2,3,Some(6)),(10,2,3,Some(0))] { assert!(HybridBudget::new(total,resident,shared,cap).is_err()); }
    let pool=HybridBudget::new(10,2,3,Some(5)).unwrap(); assert_eq!(pool.active_limit(None,5).unwrap(),1); assert!(pool.active_limit(Some(2),5).is_err()); assert_eq!(pool.active_limit(None,1).unwrap(),5);
    let l=HybridRankLedger{ceiling:200,resident:70,shared:20,transport:10,request:30,retained:11,in_flight:7,measured_fixed:100};
    let mut low=l; low.ceiling=178; let ledgers=[l,low]; let pool=HybridBudget::for_ranks(&ledgers,2,None).unwrap(); assert_eq!(pool.sessions,78); assert_eq!(HybridBudget::for_ranks(&[low,l],2,None).unwrap(),pool); assert_eq!(HybridRankLedger::from_values(l.values()),l);
    low.ceiling-=1; assert!(HybridBudget::for_ranks(&[l,low],2,None).unwrap_err().contains("rank=1")); assert!(HybridBudget::for_ranks(&ledgers,2,Some(77)).is_err()); assert!(HybridBudget::for_ranks(&ledgers,3,None).is_err());
    for (request,retained,in_flight) in [(usize::MAX,0,0),(30,usize::MAX,1),(30,1,usize::MAX)] { let mut bad=l; bad.request=request; bad.retained=retained; bad.in_flight=in_flight; assert!(HybridBudget::for_ranks(&[bad],2,None).is_err()); }
    let mut bad=l; bad.measured_fixed=101; assert!(HybridBudget::for_ranks(&[bad],1,None).is_err()); bad=l; bad.shared=usize::MAX; assert!(HybridBudget::for_ranks(&[bad],1,None).is_err()); assert!(HybridBudget::for_ranks(&[],1,None).is_err());
}
