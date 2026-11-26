use spark_gpu_probe::text_edges::{Shape,Layout,NormStorage,greedy};
#[test]
fn checks_spans_and_dynamic_stream_counts() {
    for streams in [1,2,3,4,7] {
        let shape = Shape::new(96,17,streams).unwrap(); let bytes = shape.weight_bytes().unwrap();
        let norm = bytes.iter().sum::<usize>().div_ceil(4)*4;
        let layout = Layout {shape,embedding:0,head:bytes[0],scales:bytes[0]+bytes[1],norm,norm_storage:NormStorage::F32,epsilon:1e-5};
        let size = norm+shape.hidden()*4; assert!(layout.ranges(size).is_ok());
        assert!(layout.ranges(size-1).is_err()); assert_eq!(shape.residual(),96*streams as usize);
        assert!(Layout {head:0,..layout}.ranges(size).is_err());
        assert!(Layout {head:usize::MAX,..layout}.ranges(size).is_err());
        assert!(Layout {head:layout.head+1,..layout}.ranges(size+16).is_err());
        assert!(Layout {epsilon:f32::NAN,..layout}.ranges(size).is_err());
        assert!(Layout {norm:layout.scales,..layout}.ranges(size).is_err());
        assert!(shape.token(16).is_ok()); assert!(shape.token(17).is_err());
    }
}
#[test]
fn rejects_empty_and_oversized_geometry() {
    for (h,v,s) in [(0,2,1),(31,2,1),(32,0,1),(32,2,0),(32,(1<<24)+1,1),(u64::MAX,2,1),(32,2,u32::MAX as u64)] {
        assert!(Shape::new(h,v,s).is_err());
    }
}
#[test]
fn finite_greedy_selects_lowest_id_on_ties() {
    assert_eq!(greedy(&[1.0,4.0,4.0,-2.0]).unwrap(),1);
    assert_eq!(greedy(&[-0.0,0.0]).unwrap(),0);
    for bad in [vec![],vec![f32::NAN],vec![0.0,f32::INFINITY],vec![f32::NEG_INFINITY]] {assert!(greedy(&bad).is_err());}
}
