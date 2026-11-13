//! Same softmax body as the plain step, one block per (token,head).
use crate::prompt_ops::Arguments;
#[no_mangle]
pub extern "ptx-kernel" fn prompt_softmax_rows(p:Arguments) {
    let s=p.shape;
    // Shape::lengths and softmax_rows::admitted bound all extents on the host.
    // No pointer is dereferenced here. The shared body checks every live count.
    let shape=crate::causal_ops::Shape {op:crate::causal_ops::Op::Softmax,
        heads:s.tokens*s.heads,rows:s.cols,dim:0,taps:0,cols:0,epsilon:0.0,bound:0.0};
    let args=crate::causal_ops::Arguments {input:[p.input[0],p.input[1]+s.slots as u64*4,0,0],weight:0,output:p.output,shape};
    crate::causal_fast::softmax::<true>(args,s.heads as usize,s.slots as usize+1);
}
