//! SECOND RAIL: a pair of ranks finds the second rail of its cable by itself.
//!
//! What it is for. A cable between two Sparks carries two rails. The transport already sends every pair gather of
//! `PAIR_GATHER_FLOOR_BYTES` a rank or more over both rails, half on each, but the second rail was open only when the
//! start named its two addresses (`TP_AUX_LISTEN` and `TP_AUX_NEXT`). A start that named none ran on one rail.
//! Measured on a pair (runs `link-base-x1` and `link-base-x2`, one build, one rail against two): the wire moves 13.2
//! GB/s on one rail and 23.8 GB/s on two, the link step of an 8,192-token prompt read falls from 2,788 to 1,798 ms, and
//! the read from 28,542 to 27,589 ms, with the same token ids.
//!
//! The rule. Nothing is scanned and no address is guessed. Each rank lists its own fast links other than the one the
//! pair already talks on, and the two ranks trade their lists on the link they hold. A second rail is one address of
//! each rank in one subnet. Both ranks compute the same choice from the same two lists, so both open it or neither does.
//! - A rank whose start names its second rail (`TP_AUX_LISTEN`, `TP_AUX_NEXT`: two addresses, or `off` twice for one
//!   rail) offers nothing (`Offer::Closed`). Then no rail is found by either rank, and a named rail is opened as
//!   before. The launchers write `off` for a start with `TP_DUAL_RAIL=0`.
//! - A link that is down, has no address, or runs at another speed than the pair's first link is not offered.
//! - A pair that finds no rail starts on one rail and prints one line that says why (`why_not`). A second rail that
//!   fails to open leaves the pair on one rail with one log line, as a named one does.
use std::net::Ipv4Addr;

/// Bytes of one rank's offer on the wire.
pub const OFFER_BYTES:usize=48;
/// Links one offer can hold. A Spark has four fast links; one of them is the link the pair talks on.
const SLOTS:usize=7;

/// One fast link of a rank as its own box shows it: address, prefix length, host path (the PCI domain of its device,
/// empty when unknown) and speed in Mb/s. A link that is down has no speed and is not in the list.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Link {pub address:Ipv4Addr,pub prefix:u8,pub path:String,pub mbps:u64}
/// One fast link of a rank that is not the link the pair talks on.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Candidate {
    pub address:Ipv4Addr,
    pub prefix:u8,
    /// The link reaches the host on another path than the pair's first link does (another PCI domain). Two rails on
    /// one host path share that path's rate, so a rail on the other path is taken first.
    pub apart:bool,
}
/// What one rank tells the other about a second rail.
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum Offer {
    /// This rank opens no rail the pair finds by itself.
    Closed,
    /// This rank's links, best first.
    Open(Vec<Candidate>),
}

fn network(address:Ipv4Addr,prefix:u8)->u32 {u32::from(address)&(u32::MAX<<(32-u32::from(prefix)))}
fn usable(prefix:u8)->bool {(1..=31).contains(&prefix)}

/// This rank's offer. `links`: every fast link of the rank, the pair's first link among them. `primary`: this rank's
/// address on that first link. Not offered: a link in the first link's subnet, and a link at another speed than the
/// first link (a rail that trained at fewer lanes). With no first link in the list nothing is offered: its subnet
/// and its speed are not known.
pub fn candidates(links:&[Link],primary:Ipv4Addr)->Vec<Candidate> {
    let Some(first)=links.iter().find(|link|link.address==primary && usable(link.prefix)) else {return Vec::new();};
    let own=network(primary,first.prefix);
    let mut found=links.iter().filter(|link|usable(link.prefix) && link.mbps==first.mbps && network(link.address,first.prefix)!=own && network(primary,link.prefix)!=network(link.address,link.prefix))
        .map(|link|Candidate {address:link.address,prefix:link.prefix,apart:!link.path.is_empty() && !first.path.is_empty() && link.path!=first.path}).collect::<Vec<_>>();
    found.sort_by_key(|c|(!c.apart,u32::from(c.address),c.prefix));found.dedup();found.truncate(SLOTS);found
}

/// The offer as it travels: a kind byte, a count, then six bytes a link (address, prefix length, flags).
pub fn encode(offer:&Offer)->[u8;OFFER_BYTES] {
    let mut bytes=[0u8;OFFER_BYTES];
    if let Offer::Open(links)=offer {
        bytes[0]=1;bytes[1]=links.len().min(SLOTS) as u8;
        for (slot,link) in links.iter().take(SLOTS).enumerate() {
            let at=2+slot*6;bytes[at..at+4].copy_from_slice(&link.address.octets());bytes[at+4]=link.prefix;bytes[at+5]=u8::from(link.apart);
        }
    }
    bytes
}
/// The other rank's offer. Anything that is not a well-formed open offer is `Closed`: the pair then stays on one rail.
pub fn decode(bytes:&[u8])->Offer {
    if bytes.len()!=OFFER_BYTES || bytes[0]!=1 || usize::from(bytes[1])>SLOTS || bytes[2+usize::from(bytes[1])*6..].iter().any(|&b|b!=0) {return Offer::Closed;}
    let mut links=Vec::with_capacity(usize::from(bytes[1]));
    for slot in 0..usize::from(bytes[1]) {
        let at=2+slot*6;
        if !usable(bytes[at+4]) || bytes[at+5]>1 {return Offer::Closed;}
        links.push(Candidate {address:Ipv4Addr::new(bytes[at],bytes[at+1],bytes[at+2],bytes[at+3]),prefix:bytes[at+4],apart:bytes[at+5]==1});
    }
    Offer::Open(links)
}

/// The second rail of the pair: (this rank's address, the other rank's address), or none. `rank` is 0 or 1. The two
/// ranks call this with the same two offers, swapped, and get the same rail, swapped: the choice reads the offers in
/// rank order and takes the smallest pair, a rail on the other host path at both ends first.
pub fn pick(rank:usize,own:&Offer,peer:&Offer)->Option<(Ipv4Addr,Ipv4Addr)> {
    let (Offer::Open(own),Offer::Open(peer))=(own,peer) else {return None;};
    // The lists in rank order, whichever rank asks.
    let lists=[own,peer];let (zero,one)=(*lists.get(rank)?,*lists.get(1-rank)?);
    let (a,b)=zero.iter().flat_map(|a|one.iter().map(move|b|(a,b)))
        .filter(|(a,b)|a.prefix==b.prefix && a.address!=b.address && network(a.address,a.prefix)==network(b.address,b.prefix))
        .min_by_key(|(a,b)|(!(a.apart && b.apart),u32::from(a.address),u32::from(b.address)))?;
    let ends=[a.address,b.address];Some((ends[rank],ends[1-rank]))
}

/// Why this pair stays on one rail, for the one line a start prints. Call it when `pick` found none.
pub fn why_not(own:&Offer,peer:&Offer)->&'static str {
    match (own,peer) {
        (Offer::Closed,_)=>"this rank was started for one rail",
        (_,Offer::Closed)=>"the other rank was started for one rail or names its own second rail",
        (Offer::Open(own),_) if own.is_empty()=>"this rank has no second fast link that is up, has an address and runs at the speed of the first",
        (_,Offer::Open(peer)) if peer.is_empty()=>"the other rank has no second fast link that is up, has an address and runs at the speed of the first",
        _=>"the two ranks share no second subnet",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test addresses are built from their parts (documentation ranges); no dotted address is written here.
    fn at(net:u8,host:u8)->Ipv4Addr {Ipv4Addr::new(198,51,net,host)}
    fn link(net:u8,host:u8,path:&str)->Link {Link {address:at(net,host),prefix:24,path:path.into(),mbps:200_000}}
    /// A Spark pair as cabled: one cable, two rails (subnets 1 and 2), and each rank's other port to another box.
    fn spark(host:u8,elsewhere:u8)->Vec<Link> {
        vec![link(elsewhere,9,"0000"),link(1,host,"0000"),link(elsewhere+1,9,"0002"),link(2,host,"0002")]
    }
    #[test]
    fn a_pair_finds_the_second_rail_of_its_cable_and_both_ranks_agree() {
        let (zero,one)=(Offer::Open(candidates(&spark(1,40),at(1,1))),Offer::Open(candidates(&spark(2,60),at(1,2))));
        // The first link's own subnet is never offered; the rail on the other host path is offered first.
        let Offer::Open(list)=&zero else {unreachable!()};
        assert_eq!(list[0],Candidate {address:at(2,1),prefix:24,apart:true});
        assert!(list.iter().all(|c|c.address!=at(1,1)) && list.len()==3);
        // Each rank gets its own end first, and the two answers are one rail.
        assert_eq!(pick(0,&zero,&one),Some((at(2,1),at(2,2))));
        assert_eq!(pick(1,&one,&zero),Some((at(2,2),at(2,1))));
        // The offers survive the wire unchanged.
        assert_eq!(decode(&encode(&zero)),zero);assert_eq!(decode(&encode(&one)),one);
    }
    #[test]
    fn no_rail_is_found_without_a_shared_second_subnet_or_when_a_rank_is_closed() {
        let zero=Offer::Open(candidates(&spark(1,40),at(1,1)));
        // The other rank has no address on the second rail: its other links go elsewhere.
        let lone=Offer::Open(candidates(&[link(60,9,"0000"),link(1,2,"0000"),link(61,9,"0002")],at(1,2)));
        assert_eq!(pick(0,&zero,&lone),None);assert_eq!(pick(1,&lone,&zero),None);
        // One rank opted out or named its own rail: neither rank finds one.
        let one=Offer::Open(candidates(&spark(2,60),at(1,2)));
        assert_eq!(pick(0,&zero,&Offer::Closed),None);assert_eq!(pick(1,&Offer::Closed,&zero),None);assert_eq!(pick(0,&Offer::Closed,&one),None);
        assert_eq!(decode(&encode(&Offer::Closed)),Offer::Closed);
        // Same subnet but another prefix length at the two ends, or one address at both ends: not a rail.
        let wide=Offer::Open(vec![Candidate {address:at(2,2),prefix:16,apart:true}]);
        let same=Offer::Open(vec![Candidate {address:at(2,1),prefix:24,apart:true}]);
        let narrow=Offer::Open(vec![Candidate {address:at(2,1),prefix:24,apart:true}]);
        assert_eq!(pick(0,&narrow,&wide),None);assert_eq!(pick(0,&narrow,&same),None);
        // The first link is not in the list (or has no usable prefix): its subnet is unknown, nothing is offered.
        assert!(candidates(&spark(1,40),at(7,7)).is_empty());
        assert!(candidates(&[Link {prefix:32,..link(1,1,"0000")},link(2,1,"0002")],at(1,1)).is_empty());
        // A second address in the first link's own subnet is not a second rail.
        assert!(candidates(&[link(1,1,"0000"),link(1,5,"0002")],at(1,1)).is_empty());
    }
    #[test]
    fn a_rail_at_another_speed_is_not_offered_and_every_start_on_one_rail_says_why() {
        // The second rail of the cable trained at half the lanes on one box: that box offers nothing, the pair stays on one rail.
        let slow=vec![link(1,2,"0000"),Link {mbps:100_000,..link(2,2,"0002")}];
        let (zero,one)=(Offer::Open(candidates(&spark(1,40),at(1,1))),Offer::Open(candidates(&slow,at(1,2))));
        assert_eq!(one,Offer::Open(Vec::new()));assert_eq!(pick(0,&zero,&one),None);assert_eq!(pick(1,&one,&zero),None);
        assert!(why_not(&zero,&one).starts_with("the other rank has no second fast link"));
        assert!(why_not(&one,&zero).starts_with("this rank has no second fast link"));
        // A first link that is itself the slower one takes a second rail of its own speed only.
        let both_slow=vec![Link {mbps:100_000,..link(1,1,"0000")},Link {mbps:100_000,..link(2,1,"0002")},link(3,1,"0002")];
        assert_eq!(candidates(&both_slow,at(1,1)),[Candidate {address:at(2,1),prefix:24,apart:true}]);
        // The other reasons, one each.
        let full=Offer::Open(candidates(&spark(2,60),at(1,2)));
        assert_eq!(why_not(&Offer::Closed,&full),"this rank was started for one rail");
        assert_eq!(why_not(&zero,&Offer::Closed),"the other rank was started for one rail or names its own second rail");
        let elsewhere=Offer::Open(candidates(&[link(1,2,"0000"),link(61,9,"0002")],at(1,2)));
        assert_eq!(pick(0,&zero,&elsewhere),None);assert_eq!(why_not(&zero,&elsewhere),"the two ranks share no second subnet");
    }
    #[test]
    fn two_cables_take_the_rail_on_the_other_host_path_and_the_choice_does_not_follow_list_order() {
        // Two cables between the same two boxes: subnets 1 (first link), 2, 3 and 4. Subnets 2 and 4 are on the other path.
        let box_links=|host:u8|vec![link(1,host,"0000"),link(3,host,"0000"),link(4,host,"0002"),link(2,host,"0002")];
        let (zero,one)=(candidates(&box_links(1),at(1,1)),candidates(&box_links(2),at(1,2)));
        let mut shuffled=one.clone();shuffled.reverse();
        for other in [one,shuffled] {
            assert_eq!(pick(0,&Offer::Open(zero.clone()),&Offer::Open(other.clone())),Some((at(2,1),at(2,2))));
            assert_eq!(pick(1,&Offer::Open(other),&Offer::Open(zero.clone())),Some((at(2,2),at(2,1))));
        }
        // With no host path known, any shared subnet is a rail, and the two ranks still agree.
        let plain=|host:u8|Offer::Open(candidates(&[link(1,host,""),link(3,host,""),link(2,host,"")],at(1,host)));
        assert_eq!(pick(0,&plain(1),&plain(2)),Some((at(2,1),at(2,2))));assert_eq!(pick(1,&plain(2),&plain(1)),Some((at(2,2),at(2,1))));
    }
    #[test]
    fn a_damaged_offer_is_closed_and_an_offer_never_holds_more_than_its_slots() {
        let many=(0..12).map(|i|link(10+i,1,"0002")).chain([link(1,1,"0000")]).collect::<Vec<_>>();
        let offer=Offer::Open(candidates(&many,at(1,1)));
        let Offer::Open(list)=&offer else {unreachable!()};assert_eq!(list.len(),SLOTS);
        let bytes=encode(&offer);assert_eq!(decode(&bytes),offer);
        // Every single damaged field closes the offer: kind, count, a prefix length, a flag, bytes after the last link, the length.
        for (at,value) in [(0usize,2u8),(1,8),(2+4,0),(2+4,32),(2+5,2)] {let mut bad=bytes;bad[at]=value;assert_eq!(decode(&bad),Offer::Closed,"byte {at}={value}");}
        let mut short=encode(&Offer::Open(vec![Candidate {address:at(2,1),prefix:24,apart:false}]));short[20]=1;assert_eq!(decode(&short),Offer::Closed);
        assert_eq!(decode(&bytes[..OFFER_BYTES-1]),Offer::Closed);assert_eq!(decode(&[0u8;OFFER_BYTES]),Offer::Closed);
    }
}
