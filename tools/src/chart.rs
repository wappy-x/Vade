//! Draw the release figure from committed receipt text. No values are rounded.
use std::{fs,path::Path};
use crate::Result;
const RELEASE: &str = "5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e";
const HEADLINE: &str = "A second Spark answers a long prompt faster. One chat does not write faster.";
const SUBHEAD: &str = "With drafting ON (the default), one chat writes slower on two Sparks. Eight chats vary from run to run.";
const NAMES: [&str;2] = ["1 Spark","2 Sparks"];
fn folder(time: &str, count: &str, suffix: &str) -> String {
    format!("2026-10-24_{time}_qwen3.8-flash-next-nvfp4_local-{count}_5a22651-drafting-on-thinking-off{suffix}")
}
fn card(root: &Path, name: &str) -> Result<[String;4]> {
    let path = root.join("bench/receipts/2026-10-24").join(name);
    if !fs::read_to_string(path.join("README.txt"))?.contains(RELEASE) { return Err(format!("{name}: README does not name release commit").into()); }
    let mut paths: Vec<_> = fs::read_dir(&path)?.map(|p| p.map(|p| p.path())).collect::<std::io::Result<_>>()?; paths.sort();
    let mut found: [Vec<String>;4] = std::array::from_fn(|_| Vec::new());
    for path in paths.iter().filter(|p| p.to_string_lossy().ends_with(".card.txt")) {
        for line in fs::read_to_string(path)?.lines() {
            let words: Vec<_> = line.split_whitespace().collect();
            for (i,key) in ["CODE","PROSE","C8","8K"].iter().enumerate() {
                let value = if i == 2 && line.contains("Aggregate tok/s:") { words.iter().position(|s| s == key).and_then(|p| words.get(p+1)) }
                    else if words.first() == Some(&"│") && words.get(1) == Some(key) { words.get(if i == 3 {3} else {2}) } else { None };
                if let Some(value) = value {
                    let parts: Vec<_> = value.split('.').collect();
                    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit())) || (i != 3 && parts[1].len() != 1) {
                        return Err(format!("{name}: invalid {key} value").into());
                    }
                    found[i].push((*value).into());
                }
            }
        }
    }
    if found.iter().any(|v| v.len() != 1) { return Err(format!("{name}: expected one value for each field").into()); }
    Ok(found.map(|mut v| v.remove(0)))
}
fn number(s: &str) -> f64 { s.parse().expect("receipt digits checked") }
fn spread(cards: &[[String;4]], i: usize) -> (&str,&str) {
    let mut values: Vec<_> = cards.iter().map(|c| c[i].as_str()).collect(); values.sort_by(|a,b| number(a).total_cmp(&number(b))); (values[0],values[values.len()-1])
}
fn esc(s: &str) -> String { s.replace('&',"&amp;").replace('<',"&lt;").replace('>',"&gt;") }
fn width(s: &str, size: f64, tabular: bool) -> f64 {
    s.chars().map(|c| size * if ".,:;il|() \"".contains(c) || (!tabular && c == '1') {0.28} else if "mwMW".contains(c) {0.85} else {0.58}).sum()
}
struct BoxText { l: f64, t: f64, r: f64, b: f64, text: String }
struct Drawing<'a> { out: Vec<String>, labels: Vec<BoxText>, bars: Vec<[f64;4]>, colors: [&'a str;7] }
impl Drawing<'_> {
    fn label(&mut self, x: f64, y: f64, body: &str, size: f64, color: usize, weight: Option<u32>, tabular: bool) {
        self.labels.push(BoxText { l:x,t:y-size*0.8,r:x+width(body,size,tabular),b:y+size*0.2,text:body.into() });
        let mut extra = weight.map(|w| format!(" font-weight=\"{w}\"")).unwrap_or_default();
        if tabular { extra.push_str(" font-variant-numeric=\"tabular-nums\""); }
        self.out.push(format!("<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{size}\" fill=\"{}\" text-anchor=\"start\"{extra}>{}</text>",self.colors[color],esc(body)));
    }
    fn pair(&mut self, left: f64, top: f64, values: [&str;2], max: f64, pixels: f64, unit: &str) -> Result<()> {
        let base = left+66.;
        self.out.push(format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"1\" height=\"48\" fill=\"{}\"/>",base-1.,top-3.,self.colors[4]));
        for (k,value) in values.iter().enumerate() {
            let y = top+k as f64*22.; let length = number(value)/max*pixels;
            if !(0. < length && length <= pixels) { return Err(format!("{value} is outside the drawn scale").into()); }
            let r = 4f64.min(length/2.); let end = base+length; let bottom = y+20.;
            self.out.push(format!("<path d=\"M{base:.1} {y:.1}H{:.1}Q{end:.1} {y:.1} {end:.1} {:.1}V{:.1}Q{end:.1} {bottom:.1} {:.1} {bottom:.1}H{base:.1}Z\" fill=\"{}\"/>",end-r,y+r,bottom-r,end-r,self.colors[k+5]));
            self.bars.push([base,y,end,bottom]);
            self.label(left,y+14.5,NAMES[k],12.,2,None,false);
            self.label(end+6.,y+14.5,&format!("{value}{unit}"),12.,2,None,true);
        } Ok(())
    }
    fn validate(&self, panel_bottom: f64) -> Result<()> {
        for (i,a) in self.labels.iter().enumerate() {
            if a.l < 23.5 || a.r > 856.5 { return Err(format!("text runs past the margin: {}",a.text).into()); }
            if a.l < 572. && a.r > 560. && a.t < panel_bottom && a.b > 114. { return Err(format!("text runs into the right panel: {}",a.text).into()); }
            for b in &self.labels[i+1..] {
                if a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b { return Err(format!("labels collide: {} / {}",a.text,b.text).into()); }
            }
            for &[l,t,r,b] in &self.bars {
                if a.l < r+1. && l-1. < a.r && a.t < b && t < a.b { return Err(format!("label touches a bar: {}",a.text).into()); }
            }
        } Ok(())
    }
}
pub fn render(theme: &str, data: &[[String;4];2], note: &[String;2], folders: &[String;2]) -> Result<String> {
    let colors = if theme == "light" { ["#fcfcfb","rgba(11,11,11,0.10)","#0b0b0b","#52514e","#c3c2b7","#2a78d6","#eb6834"] }
        else { ["#1a1a19","rgba(255,255,255,0.10)","#f0efec","#c3c2b7","#383835","#3987e5","#d95926"] };
    let mut d = Drawing { out:Vec::new(), labels:Vec::new(), bars:Vec::new(), colors };
    d.label(24.,40.,HEADLINE,19.,2,Some(650),false); d.label(24.,63.,SUBHEAD,13.,3,None,false);
    let mut x = 24.;
    for (i,name) in NAMES.iter().enumerate() {
        d.out.push(format!("<rect x=\"{x}\" y=\"80\" width=\"12\" height=\"12\" rx=\"2\" fill=\"{}\"/>",colors[i+5]));
        d.label(x+18.,90.5,name,12.5,2,Some(600),false); x += 18.+width(name,12.5,false)+22.;
    }
    d.label(x+4.,90.5,"Vade v0.1.0 on our own DGX Sparks, drafting ON",12.,3,None,false);
    d.label(24.,128.,"Writing speed",14.,2,Some(600),false);
    d.label(24.,146.,"tokens per second, higher is better",12.,3,None,false);
    d.label(572.,128.,"First reply after an 8K-token prompt",14.,2,Some(600),false);
    d.label(572.,146.,"seconds, lower is better",12.,3,None,false);
    let mut y = 174.;
    for (i,title) in ["Code, 1 chat","Prose, 1 chat","8 chats together"].iter().enumerate() {
        d.label(24.,y,title,12.5,2,Some(600),false); d.pair(24.,y+9.,[&data[0][i],&data[1][i]],180.,376.,"")?;
        y += 51.; if i == 2 { y += 4.; for line in note { y += 16.; d.label(24.,y,line,11.5,3,None,false); } } y += 30.;
    }
    let panel_bottom = y-30.; d.pair(572.,170.5,[&data[0][3],&data[1][3]],4.5,190.," s")?; y += 2.;
    for line in ["Each bar is one RigMark card for Vade v0.1.0 (5a22651), thinking OFF, context 32768, greedy decoding.",
        "Two Sparks run as a pipeline: each holds part of the model's layers and passes its work to the next.",
        "The headline holds on all six single Sparks and all three pairs we measured with these settings.",
        "Drafting OFF and the cloud VM runs are in the tables on the benchmarks page.","Sources, bench/receipts/2026-10-24/:"] {
        d.label(24.,y,line,11.,3,None,false); y += 17.;
    }
    for folder in folders { d.label(36.,y,folder,10.5,3,None,false); y += 15.; }
    d.validate(panel_bottom)?;
    let height = (y+8.) as u32; let [one,two] = data;
    let desc = format!("{HEADLINE} {SUBHEAD} Drafting ON, 1 Spark then 2 Sparks: code {} and {} tok/s, prose {} and {}, 8 chats together {} and {}; first reply after an 8K-token prompt {} s and {} s. {} {}",one[0],two[0],one[1],two[1],one[2],two[2],one[3],two[3],note[0],note[1]);
    Ok(format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 880 {height}\" width=\"880\" height=\"{height}\" role=\"img\" aria-labelledby=\"t d\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">\n<title id=\"t\">Vade v0.1.0 on one DGX Spark and on two, drafting ON, RigMark cards of 2026-10-24</title>\n<desc id=\"d\">{}</desc>\n<rect x=\"0.5\" y=\"0.5\" width=\"879\" height=\"{}\" rx=\"8\" fill=\"{}\" stroke=\"{}\"/>\n{}\n</svg>\n",esc(&desc),height-1,colors[0],colors[1],d.out.join("\n")))
}
pub fn run(root: &Path) -> Result<()> {
    let folders = [folder("1943","1spark",""),folder("2019","2spark","")];
    let singles = [("1943",""),("1945",""),("1946",""),("1946","-home-5"),("1947",""),("1947","-home-6")]
        .iter().map(|(t,s)| card(root,&folder(t,"1spark",s))).collect::<Result<Vec<_>>>()?;
    let pairs = [("2019",""),("2022",""),("2022","-home-5-home-6")].iter().map(|(t,s)| card(root,&folder(t,"2spark",s))).collect::<Result<Vec<_>>>()?;
    if number(spread(&pairs,3).1) >= number(spread(&singles,3).0) || (0..2).any(|i| number(spread(&pairs,i).1) >= number(spread(&singles,i).0)) {
        return Err("the cards no longer support the headline; update the headline and subtitle".into());
    }
    let (a,b) = spread(&singles,2); let (c,d) = spread(&pairs,2);
    let note = [format!("Across all our runs, 8 chats gave {a} to {b} tok/s"),format!("on six single Sparks and {c} to {d} on three pairs.")];
    let data = [card(root,&folders[0])?,card(root,&folders[1])?];
    for theme in ["light","dark"] {
        let path = root.join(format!("docs/assets/benchmarks-2026-10-24-{theme}.svg"));
        fs::write(&path,render(theme,&data,&note,&folders)?)?; println!("wrote {}",path.strip_prefix(root)?.display());
    } Ok(())
}
