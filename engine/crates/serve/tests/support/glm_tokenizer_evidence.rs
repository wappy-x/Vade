//! Metadata inventory and fixed publisher values. No BPE or split reference runs here.
use super::{artifact_root, checkpoint, sha256};
use engine_model::json::{self, Value};
use spark_serve::Tokenizer as _;
use std::collections::BTreeMap;

const PUBLISHER: &[u8] = include_bytes!("../fixtures/glm-tokenizer/publisher-values.json");
const SOURCE: &str = include_str!("../fixtures/glm-tokenizer/nvidia-glm-frontend.rs.txt");
const CONTRACT: &[u8] = include_bytes!("../fixtures/glm-tokenizer/nvidia-glm53.json");
const HEADERS: &[u8] = include_bytes!("../fixtures/glm-tokenizer/row-headers.json");
fn parse(bytes: &[u8]) -> Value {
    json::parse(bytes,json::Limits {bytes:32<<20,nodes:2_000_000,string_bytes:1<<20,depth:24}).unwrap()
}

#[test]
fn publisher_values_retain_the_pinned_expectations() {
    let fixture=parse(PUBLISHER); let f=fixture.object().unwrap();
    assert_eq!(sha256::digest(SOURCE.as_bytes()),f["source_sha256"].string().unwrap());
    assert_eq!(sha256::digest(CONTRACT),"8b1204a0db52a35dc8f984dffb36419c4f58374cb32d082a40c076bdc2540825");
    assert!(SOURCE.contains("eos_token_ids: vec![154820, 154827, 154829]"));
    assert!(SOURCE.contains("(154841, \"<think>\".to_owned())"));
    assert!(SOURCE.contains("(154842, \"</think>\".to_owned())"));
    assert!(SOURCE.contains("#[ignore = \"requires FLARE_GLM_TEST_SNAPSHOT and FLARE_GLM_FRONTEND_REFERENCE"));
    assert_eq!(f["source_revision"].string().unwrap(),"18b16e170bb9eab03e2b1800107c72966c56594f");
    assert_eq!(f["checkpoint_revision"].string().unwrap(),"76348ddb2be2e35e39ecfc00e75dc07703f4f3d5");
    let contract=parse(CONTRACT);
    assert_eq!(contract.object().unwrap()["vocabulary"].u64().unwrap(),row_count());
}
fn row_count() -> u64 {
    assert_eq!(sha256::digest(HEADERS),"86332684a9ccc25ab203bf54b6f530827ab5dd16a79b27948805f8f7a462a0d5");
    let headers=parse(HEADERS); let headers=headers.object().unwrap();
    let rows=headers["lm_head.weight"].object().unwrap()["shape"].array().unwrap()[0].u64().unwrap();
    for name in ["lm_head.weight_scale","model.language_model.embed_tokens.weight"] {
        assert_eq!(headers[name].object().unwrap()["shape"].array().unwrap()[0].u64().unwrap(),rows);
    }
    let config=parse(include_bytes!("../../../model/tests/fixtures/glm/config.json"));
    assert_eq!(config.object().unwrap()["text_config"].object().unwrap()["vocab_size"].u64().unwrap(),rows);
    rows
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn publisher_delimiter_ids_decode_bytes_and_eos_match_the_checkpoint() {
    let model=checkpoint(); let fixture=parse(PUBLISHER); let fixture=fixture.object().unwrap();
    let tokenizer=std::fs::read(artifact_root().join("tokenizer.json")).unwrap();
    assert_eq!(sha256::digest(&tokenizer),fixture["tokenizer_sha256"].string().unwrap());
    for value in fixture["delimiters"].array().unwrap() {
        let fields=value.object().unwrap(); let id=u32::try_from(fields["id"].u64().unwrap()).unwrap();
        let expected=fields["decoded"].string().unwrap();
        assert_eq!(model.token_bytes(id).unwrap(),expected.as_bytes());
        assert_eq!(model.decode(&[id],false).unwrap().as_bytes(),expected.as_bytes());
        assert_eq!(engine_tokenizer::Tokenizer::encode(&model,expected).unwrap(),[id]);
        let mut stream=model.stream_decoder(false);
        assert_eq!((stream.push(id).unwrap()+&stream.finish()).as_bytes(),expected.as_bytes());
    }
    let expected=fixture["eos_ids"].array().unwrap().iter().map(|v|u32::try_from(v.u64().unwrap()).unwrap()).collect::<Vec<_>>();
    assert_eq!(model.checkpoint_eos_token_ids().unwrap().unwrap(),expected);
    println!("PUBLISHER VALUES: delimiter_pairs=2 eos_ids={expected:?}; no upstream runtime was executed");
}

struct Inventory { rows: Vec<Option<(Vec<u8>,bool)>>, vocab: usize, added: usize, special: usize }
// This reads byte spellings only. It does not encode text, split text, or apply merges.
// JSON member order has no row meaning. Explicit IDs set the row order.
fn inventory(root: &Value, row_count: usize) -> Result<Inventory,String> {
    let object=root.object().map_err(|e|e.to_string())?;
    let model=object["model"].object().map_err(|e|e.to_string())?;
    let vocab=model["vocab"].object().map_err(|e|e.to_string())?;
    let added=object["added_tokens"].array().map_err(|e|e.to_string())?;
    let mut inverse=BTreeMap::new(); let mut next=256;
    for b in 0..=255u32 {
        let c=if (33..=126).contains(&b)||(161..=172).contains(&b)||(174..=255).contains(&b) {b}
            else {next+=1;next-1};
        inverse.insert(char::from_u32(c).unwrap(),b as u8);
    }
    let mut rows=vec![None;row_count];
    for (text,value) in vocab {
        let id=usize::try_from(value.u64().map_err(|e|e.to_string())?).map_err(|_|"row ID overflow")?;
        if id>=vocab.len() || id>=rows.len() {return Err("base vocabulary is not a dense row prefix".into());}
        if rows[id].is_some() {return Err("duplicate vocabulary row".into());}
        let bytes=text.chars().map(|c|inverse.get(&c).copied().ok_or("invalid byte spelling"))
            .collect::<Result<Vec<_>,_>>()?;
        rows[id]=Some((bytes,false));
    }
    let mut special=0;
    for (index,value) in added.iter().enumerate() {
        let fields=value.object().map_err(|e|e.to_string())?;
        let id=usize::try_from(fields["id"].u64().map_err(|e|e.to_string())?).map_err(|_|"row ID overflow")?;
        if id>=rows.len() {return Err("added token row exceeds tensor rows".into());}
        if rows[id].is_some() {return Err("added token row collides with another row".into());}
        if id!=vocab.len()+index {return Err("added token rows are not in source ID order".into());}
        let text=fields["content"].string().map_err(|e|e.to_string())?;
        let flag=fields["special"].boolean().map_err(|e|e.to_string())?;
        special+=usize::from(flag); rows[id]=Some((text.as_bytes().to_vec(),flag));
    }
    Ok(Inventory {rows,vocab:vocab.len(),added:added.len(),special})
}

#[test]
fn evidence_inventory_refuses_incomplete_or_conflicting_row_claims() {
    // These refusals belong to the evidence inventory. The general loader permits sparse IDs.
    let source=r#"{"model":{"vocab":{"!":0,"a":1}},"added_tokens":[{"id":2,"content":"<x>","special":true},{"id":3,"content":"<y>","special":false}]}"#;
    assert!(inventory(&parse(source.as_bytes()),5).is_ok());
    for (from,to,reason) in [
        (r#""a":1"#,r#""a":0"#,"duplicate vocabulary row"),
        (r#""a":1"#,r#""a":4"#,"base vocabulary is not a dense row prefix"),
        (r#""id":2"#,r#""id":1"#,"added token row collides with another row"),
        (r#""id":3"#,r#""id":5"#,"added token row exceeds tensor rows"),
        (r#""id":2"#,r#""id":4"#,"added token rows are not in source ID order"),
    ] {
        let changed=source.replacen(from,to,1); assert_ne!(changed,source);
        assert_eq!(inventory(&parse(changed.as_bytes()),5).err().unwrap(),reason);
    }
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn complete_vocabulary_rows_and_decode_bytes_match_source_metadata() {
    let model=checkpoint(); let bytes=std::fs::read(artifact_root().join("tokenizer.json")).unwrap();
    let source=parse(&bytes); let count=usize::try_from(row_count()).unwrap();
    let inventory=inventory(&source,count).unwrap(); let mapped=inventory.vocab+inventory.added;
    assert_eq!((inventory.vocab,inventory.added,inventory.special,count),(154820,36,18,154880));
    let mut receipt=Vec::new(); let mut decode_receipt=Vec::new(); let mut invalid_utf8=0;
    for (id,row) in inventory.rows.iter().enumerate() {
        let Some((bytes,special))=row else {
            assert!(!model.contains_id(id as u32));
            assert!(model.token_bytes(id as u32).is_err());
            assert!(model.decode(&[id as u32],false).unwrap_err().0.contains("unknown token ID"));
            continue;
        };
        assert_eq!(model.token_bytes(id as u32).unwrap(),bytes,"row {id}");
        assert_eq!(model.is_special(id as u32),*special,"row {id}");
        let decoded=String::from_utf8_lossy(bytes); invalid_utf8+=usize::from(std::str::from_utf8(bytes).is_err());
        assert_eq!(model.decode(&[id as u32],false).unwrap().as_bytes(),decoded.as_bytes(),"row {id}");
        assert_eq!(model.decode(&[id as u32],true).unwrap().as_bytes(),if *special {b""} else {decoded.as_bytes()},"row {id}");
        let mut stream=model.stream_decoder(false);
        assert_eq!((stream.push(id as u32).unwrap()+&stream.finish()).as_bytes(),decoded.as_bytes(),"row {id}");
        receipt.extend_from_slice(&(id as u32).to_le_bytes()); receipt.push(u8::from(*special));
        receipt.extend_from_slice(&(bytes.len() as u32).to_le_bytes()); receipt.extend_from_slice(bytes);
        decode_receipt.extend_from_slice(&(id as u32).to_le_bytes());
        decode_receipt.extend_from_slice(&(decoded.len() as u32).to_le_bytes()); decode_receipt.extend_from_slice(decoded.as_bytes());
    }
    assert!(model.token_bytes(count as u32).is_err());
    assert_eq!(invalid_utf8,1095);
    assert_eq!(sha256::digest(&receipt),"bedb1bc06709b65c1b3565067c6439c520f81ced2167766fa9fbb588746ae1c4");
    assert_eq!(sha256::digest(&decode_receipt),"d3a07fb8ec56532686a5e458a6e35d65d6ee2d4861640318848a58d685d6e390");
    println!("SOURCE INVENTORY: vocab={} added={} special={} mapped={mapped} tensor_rows={count} unmapped={} invalid_utf8_rows={invalid_utf8} row_sha256={}",
        inventory.vocab,inventory.added,inventory.special,count-mapped,sha256::digest(&receipt));
    let model_source=source.object().unwrap()["model"].object().unwrap();
    assert_eq!(model_source["merges"].array().unwrap().len(),321649);
    assert!(model_source["ignore_merges"].boolean().unwrap());
    println!("SOURCE DECODE: row_sha256={} merges={} ignore_merges={}",sha256::digest(&decode_receipt),
        model_source["merges"].array().unwrap().len(),model_source["ignore_merges"].boolean().unwrap());
    for value in source.object().unwrap()["added_tokens"].array().unwrap() {
        let f=value.object().unwrap(); println!("ADDED ROW: id={} special={} text={:?}",f["id"].u64().unwrap(),f["special"].boolean().unwrap(),f["content"].string().unwrap());
    }
}
