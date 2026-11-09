use std::collections::BTreeMap;

/// One sale: the region and the amount in cents.
struct Row {
    region: String,
    amount: u64,
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: tally FILE");
    let text = std::fs::read_to_string(&path).expect("read the sales file");
    let rows: Vec<Row> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut fields = l.split(',');
            let region = fields.next().unwrap().trim().to_string();
            let amount = fields.next().unwrap().trim().parse().unwrap();
            Row { region, amount }
        })
        .collect();
    let mut by_region: BTreeMap<&str, u64> = BTreeMap::new();
    for r in &rows {
        *by_region.entry(&r.region).or_default() += r.amount;
    }
    for (region, amount) in &by_region {
        println!("{region} {amount}");
    }
    let total: u32 = rows.iter().map(|r| r.amount).sum();
    println!("rows {} total {total}", rows.len());
}
