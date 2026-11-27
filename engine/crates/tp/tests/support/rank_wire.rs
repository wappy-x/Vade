use std::{sync::mpsc::{channel, Sender}, time::Duration};

/// Each rank reads the other ranks' messages in origin order, as in the routed service.
struct RankWire {
    topology: Topology,
    controller: bool,
    peers: Vec<Peer>,
}

struct Peer {
    origin: usize,
    send: Sender<Vec<u8>>,
    receive: Receiver<Vec<u8>>,
}

impl RankWire {
    fn group(parts: u32) -> impl Iterator<Item = Self> {
        let mut ranks = (0..parts).map(|rank| Self {
            topology: Topology {rank,parts}, controller: false, peers: Vec::new(),
        }).collect::<Vec<_>>();
        ranks[0].controller = true;
        for origin in 0..ranks.len() {
            for target in origin+1..ranks.len() {
                let (out,incoming)=channel();
                let (reply,back)=channel();
                ranks[origin].peers.push(Peer { origin: target, send: out, receive: back });
                ranks[target].peers.push(Peer { origin, send: reply, receive: incoming });
            }
        }
        ranks.into_iter()
    }

    fn send(peer: &Sender<Vec<u8>>, bytes: Vec<u8>) -> Result<()> {
        peer.send(bytes).map_err(|e| e.to_string())
    }

    fn receive(peer: &Receiver<Vec<u8>>) -> Result<Vec<u8>> {
        peer.recv_timeout(Duration::from_secs(20)).map_err(|e| e.to_string())
    }
}

impl Exchange for RankWire {
    fn topology(&self) -> Topology {self.topology}

    fn gather(&mut self, local: &[f32], rows: usize) -> Result<Vec<f32>> {
        let bytes = local.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        for peer in &self.peers {Self::send(&peer.send,bytes.clone())?;}
        let rank = self.topology.rank as usize;
        let parts = self.topology.parts as usize;
        let mut gathered = vec![0.0;rows];
        gathered[spark_tp::rows(rows,rank,parts)].copy_from_slice(local);
        for peer in &self.peers {
            let target = &mut gathered[spark_tp::rows(rows,peer.origin,parts)];
            let bytes = Self::receive(&peer.receive)?;
            assert_eq!(bytes.len(),target.len()*4);
            for (word,slot) in bytes.as_chunks::<4>().0.iter().zip(target) {
                *slot=f32::from_le_bytes(*word);
            }
        }
        Ok(gathered)
    }

    fn command(&mut self, bytes: &mut [u8]) -> Result<()> {
        if self.controller {
            for peer in &self.peers {Self::send(&peer.send,bytes.to_vec())?;}
        } else {
            let received = Self::receive(&self.peers[0].receive)?;
            bytes.copy_from_slice(&received);
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<()> {
        for peer in &self.peers {Self::send(&peer.send,Vec::new())?;}
        for peer in &self.peers {assert!(Self::receive(&peer.receive)?.is_empty());}
        Ok(())
    }
}
