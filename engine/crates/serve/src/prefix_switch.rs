//! The Flash-Next prefix cache's startup switches (FlashNext reads them; flashnext builds on
//! Linux only, so they live here, host-tested) and the cache's topology guard.

/// VADE_FLASHNEXT_PREFIX_CACHE: 1 = on (requests with `cache:true` or a `prompt_cache_key`
/// retain their prompt); 0 = off (no snapshots); unset = on where a prefix restore exists
/// (`supported`: one Spark or the tensor split, on two Sparks or in parts on four), off on the
/// layer chain. An explicit 1 there is refused: the chain has no prefix restore contract.
pub fn prefix_cache(supported: bool) -> Result<bool, String> {
    resolve(setting(PREFIX_CACHE.0, std::env::var(PREFIX_CACHE.0))?, supported)
}
fn resolve(setting: Option<bool>, supported: bool) -> Result<bool, String> {
    match setting {
        Some(true) if !supported => Err("VADE_FLASHNEXT_PREFIX_CACHE=1 needs one Spark or the tensor split: the layer chain has no prefix restore contract; unset it or set 0 there".into()),
        Some(on) => Ok(on),
        None => Ok(supported),
    }
}
/// The automatic room is half the pool bytes no full-context chat can use, at least this...
pub const ROOM_FLOOR: usize = 512 << 20;
/// ...and at most this, and never more than those bytes.
pub const ROOM_CAP: usize = 4 << 30;
/// VADE_FLASHNEXT_CACHE_ROOM: bytes the retained cache keeps as its own room (0 = none, the
/// cache uses only memory live chats leave free); unset = automatic (cache_room).
pub fn cache_room_setting() -> Result<Option<usize>, String> {
    room_setting(std::env::var(CACHE_ROOM))
}
const CACHE_ROOM: &str = "VADE_FLASHNEXT_CACHE_ROOM";
fn room_setting(value: Result<String, std::env::VarError>) -> Result<Option<usize>, String> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(text) if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) =>
            text.parse().map(Some).map_err(|_| format!("{CACHE_ROOM} is larger than this host's address space")),
        _ => Err(format!("{CACHE_ROOM} must be a byte count (0 = no room)")),
    }
}
/// The retained cache's own room, taken from the session pool at startup. Admission does
/// not use it and host memory pressure does not reclaim it; the cache still also uses pool
/// bytes live chats leave free, and yields those first. `pool` funds `active` chat slots of
/// up to `reservation` bytes (one full-context chat). Automatic: half of the bytes left after
/// the full-context chats the pool funds, clamped to ROOM_FLOOR..=ROOM_CAP, and never above
/// those bytes, so the room never lowers the number of full-context chats. On the tensor
/// split (`split`) each rank keeps a copy of every kept prompt, so the automatic room is at
/// least one full-context chat (the bytes of the longest prompt's copy), up to ROOM_CAP; it
/// may lower the number of full-context chats. A setting is capped so that one full-context
/// chat still fits; so is the split's room.
pub fn cache_room(setting: Option<usize>, pool: usize, active: usize, reservation: usize, split: bool) -> usize {
    let full = active.min(pool.checked_div(reservation).unwrap_or(active));
    let left = pool.saturating_sub(full.saturating_mul(reservation));
    let most = pool.saturating_sub(reservation);
    match setting {
        Some(bytes) => bytes.min(most),
        None if split => (left / 2).max(reservation).clamp(ROOM_FLOOR, ROOM_CAP).min(most),
        None => (left / 2).clamp(ROOM_FLOOR, ROOM_CAP).min(left),
    }
}
/// Bytes the startup slot plan keeps above one full-context chat when the cache is on: the
/// room floor, or the set room. A start whose free memory falls just above a graph-width
/// step would otherwise take the wider bank and leave the room almost nothing. 0 = the plan
/// as before (cache off, or a room of 0).
pub fn plan_room(prefix_cache: bool, setting: Option<usize>) -> usize {
    if prefix_cache { setting.unwrap_or(ROOM_FLOOR) } else { 0 }
}
/// The retained-bytes ceiling: the room plus half the pool admission uses (as before).
pub fn retained_ceiling(pool: usize, room: usize) -> usize {
    room + pool.saturating_sub(room) / 2
}
/// VADE_FLASHNEXT_CACHE_YIELD: unset or 1 = retained prefixes yield to current host memory
/// pressure 30 s after the last live request (SessionCache::rebalance_within); 0 = the session
/// pool alone bounds the cache.
pub fn cache_yield() -> Result<bool, String> {
    switch(CACHE_YIELD, std::env::var(CACHE_YIELD.0))
}
/// Unused reserve capacity owns no memory. Actual spare buffers already count in `room`.
/// Retiring a request moves its buffers to the reserve without a new allocation.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn cache_physical_room(reserve: Option<(i128, usize)>, slack: usize) -> Option<(i128, usize)> {
    reserve.map(|(room, _unused_capacity)| (room, slack))
}
/// Independent memories must each have room for a retained prefix.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn cache_rank_room(local: Option<(i128, usize)>, peer: Option<i128>, slack: usize) -> Option<(i128, usize)> {
    match (local, peer) {
        (Some((room, margin)), Some(peer)) => Some((room.min(peer), margin.max(slack))),
        (None, Some(peer)) => Some((peer, slack)),
        (local, None) => local,
    }
}
/// VADE_FLASHNEXT_ADOPT: 1 = a finished request's KV arenas stay with the prompt prefix it
/// published in place (no row copies), and the conversation's next turn builds its session
/// on them (next_prefix.rs); 0 = copies. Unset = in place on one Spark (`local`); copies on the
/// tensor split, whose back rank always copies. Needs the prefix cache.
pub fn adopt(local: bool) -> Result<bool, String> {
    adopt_from(std::env::var(ADOPT.0), local)
}
fn adopt_from(value: Result<String, std::env::VarError>, local: bool) -> Result<bool, String> {
    Ok(setting(ADOPT.0, value)?.unwrap_or(local))
}
/// The switches and their values when unset (the prefix cache: where supported; adoption: on
/// for one Spark only, see adopt).
const PREFIX_CACHE: (&str, bool) = ("VADE_FLASHNEXT_PREFIX_CACHE", true);
const CACHE_YIELD: (&str, bool) = ("VADE_FLASHNEXT_CACHE_YIELD", true);
const ADOPT: (&str, bool) = ("VADE_FLASHNEXT_ADOPT", false);
/// A 0/1 switch: its default when unset; any other value, a non-Unicode one included, refused.
fn switch((name, unset): (&str, bool), value: Result<String, std::env::VarError>) -> Result<bool, String> {
    Ok(setting(name, value)?.unwrap_or(unset))
}
fn setting(name: &str, value: Result<String, std::env::VarError>) -> Result<Option<bool>, String> {
    match value.as_deref() {
        Ok("0") => Ok(Some(false)),
        Ok("1") => Ok(Some(true)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        _ => Err(format!("{name} must be 0 or 1")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::VarError;

    #[test]
    fn physical_room_keeps_pressure_and_the_disabled_reserve() {
        assert_eq!(cache_physical_room(None, 1024), None);
        for unused in [0, usize::MAX] {
            assert_eq!(cache_physical_room(Some((-1, unused)), 1024), Some((-1, 1024)));
            assert_eq!(cache_physical_room(Some((4096, unused)), 1024), Some((4096, 1024)));
        }
        assert_eq!(cache_rank_room(None, None, 1024), None);
        assert_eq!(cache_rank_room(None, Some(-1), 1024), Some((-1, 1024)));
        assert_eq!(cache_rank_room(Some((4096, 2048)), Some(1024), 512), Some((1024, 2048)));
        assert_eq!(cache_rank_room(Some((-1, 512)), Some(4096), 1024), Some((-1, 1024)));
    }

    #[test]
    fn switches_take_zero_or_one_and_the_cache_defaults_on() {
        // Shipped defaults: the cache on where supported, adoption off on the split, yielding on.
        assert!(PREFIX_CACHE.1 && !ADOPT.1 && CACHE_YIELD.1);
        for s in [PREFIX_CACHE, CACHE_YIELD, ADOPT] {
            let set = |v: &str| switch(s, Ok(v.to_string()));
            assert_eq!(switch(s, Err(VarError::NotPresent)), Ok(s.1));
            assert_eq!(set("0"), Ok(false));
            assert_eq!(set("1"), Ok(true));
            let refused = Err(format!("{} must be 0 or 1", s.0));
            for bad in ["", "2", "on", "true", "01", " 1", "1 "] {
                assert_eq!(set(bad), refused);
            }
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let raw = std::ffi::OsString::from_vec(vec![b'1', 0xff]);
                assert_eq!(switch(s, Err(VarError::NotUnicode(raw))), refused);
            }
        }
    }

    #[test]
    fn adoption_defaults_on_for_one_spark_and_off_on_the_split() {
        for local in [false, true] {
            assert_eq!(adopt_from(Err(VarError::NotPresent), local), Ok(local));
            assert_eq!(adopt_from(Ok("0".into()), local), Ok(false));
            assert_eq!(adopt_from(Ok("1".into()), local), Ok(true));
            assert!(adopt_from(Ok("yes".into()), local).is_err());
        }
    }

    #[test]
    fn unset_follows_the_topology_and_the_layer_chain_refuses_an_explicit_on() {
        for supported in [false, true] {
            assert_eq!(resolve(Some(false), supported), Ok(false));
            assert_eq!(resolve(None, supported), Ok(supported));
        }
        assert_eq!(resolve(Some(true), true), Ok(true));
        let refusal = resolve(Some(true), false).unwrap_err();
        assert!(refusal.contains("VADE_FLASHNEXT_PREFIX_CACHE=1") && refusal.contains("layer chain"), "{refusal}");
    }

    #[test]
    fn room_setting_takes_a_byte_count() {
        assert_eq!(room_setting(Err(VarError::NotPresent)), Ok(None));
        assert_eq!(room_setting(Ok("0".into())), Ok(Some(0)));
        assert_eq!(room_setting(Ok("1073741824".into())), Ok(Some(1 << 30)));
        for bad in ["", "-1", "1G", " 1", "0x10", "99999999999999999999999"] {
            assert!(room_setting(Ok(bad.into())).is_err(), "{bad}");
        }
    }

    #[test]
    fn automatic_room_keeps_every_full_context_chat() {
        // Pools and reservations from flashnext ready lines at 285fa8a: one Spark at context
        // 32768, two Sparks at 32768, two Sparks at 131072. The one-Spark rule (split false);
        // the split's own rule is in the_split_room_holds_one_full_context_copy.
        for (pool, active, reservation, room) in [
            (18_916_909_138, 8, 2_276_963_652, ROOM_FLOOR),
            (21_162_035_384, 8, 2_276_963_652, 1_473_163_084),
            (21_332_751_391, 2, 8_165_864_772, 2_500_510_923),
            // Slots funded by the smallest chat (more slots than full chats): 9 full chats stay.
            (21_162_035_384, 16, 2_276_963_652, ROOM_FLOOR),
        ] {
            assert_eq!(cache_room(None, pool, active, reservation, false), room);
            let full = |pool: usize| active.min(pool / reservation);
            assert_eq!(full(pool - room), full(pool));
        }
        // Less left than the floor: all of it, never a chat's bytes.
        assert_eq!(cache_room(None, 8 * 1000 + 300, 8, 1000, false), 300);
        assert_eq!(cache_room(None, 8 * 1000, 8, 1000, false), 0);
        // The cap.
        assert_eq!(cache_room(None, 64 << 30, 1, 1 << 30, false), ROOM_CAP);
        // A setting may take chats, but one full-context chat always fits.
        assert_eq!(cache_room(Some(0), 21_162_035_384, 8, 2_276_963_652, false), 0);
        assert_eq!(cache_room(Some(4 << 30), 21_162_035_384, 8, 2_276_963_652, false), 4 << 30);
        assert_eq!(cache_room(Some(usize::MAX), 5000, 2, 2000, false), 3000);
        // Whenever pool bytes remain after the full-context chats, the automatic room is funded.
        for reservation in [1000, 2_276_963_652, 8_165_864_772] {
            for active in [1, 2, 8, 16] {
                for left in [1, 999, ROOM_FLOOR - 1, ROOM_FLOOR, 2 * ROOM_CAP + 1] {
                    let pool = active * reservation + left;
                    let room = cache_room(None, pool, active, reservation, false);
                    assert!(room > 0 && room <= left, "pool {pool} active {active}: room {room}");
                }
            }
        }
        // One Spark at context 262144 (a flashnext ready line at 250fb49): one full-context chat
        // takes all but 74,037,750 pool bytes, so neither the automatic room nor a setting can
        // hold a copied 200K-token snapshot (12,274,478,925 bytes) beside that chat.
        let (pool, reservation) = (16_091_770_682, 16_017_732_932);
        assert_eq!(cache_room(None, pool, 16, reservation, false), 74_037_750);
        assert_eq!(cache_room(Some(13_000_000_000), pool, 16, reservation, false), 74_037_750);
        assert_eq!(retained_ceiling(pool, 74_037_750), 8_082_904_216);
        assert_eq!(retained_ceiling(18_916_909_138, ROOM_FLOOR), 9_726_890_025);
        assert_eq!(retained_ceiling(100, 0), 50);
    }

    #[test]
    fn the_split_room_holds_one_full_context_copy() {
        // Two Sparks at context 32768 (a flashnext ready line at ba3bbf7): half the bytes left
        // after 8 full chats gave a room of 1,014,641,958, below the 32K copies a rank keeps
        // (2,117,492,300 to 2,128,995,172 bytes), so after the card no 32K turn was kept.
        let (pool, active, reservation) = (20_244_993_133, 16, 2_276_963_652);
        assert_eq!(cache_room(None, pool, active, reservation, false), 1_014_641_958);
        let room = cache_room(None, pool, active, reservation, true);
        assert_eq!(room, reservation);
        assert!(room > 2_128_995_172);
        // At context 131072 one full chat is above the cap; one full-context chat always fits.
        assert_eq!(cache_room(None, 21_332_751_391, 2, 8_165_864_772, true), ROOM_CAP);
        assert_eq!(cache_room(None, 5000, 2, 2000, true), 3000);
        // A setting is the same on both topologies.
        for split in [false, true] {
            assert_eq!(cache_room(Some(0), pool, active, reservation, split), 0);
            assert_eq!(cache_room(Some(4 << 30), pool, active, reservation, split), 4 << 30);
        }
    }

    #[test]
    fn the_slot_plan_keeps_the_room_floor_above_one_full_chat() {
        // An in-place entry is a chat's arena plus about 197 MB of its own state (a 128K prompt at
        // context 262144 at 250fb49: 16,215,083,690 retained bytes against a 16,056,182,176 pool).
        assert_eq!(plan_room(true, None), ROOM_FLOOR);
        assert!(plan_room(true, None) > 197_350_758);
        assert_eq!(plan_room(true, Some(4 << 30)), 4 << 30);
        // No cache or no room: the plan as before.
        assert_eq!(plan_room(true, Some(0)), 0);
        assert_eq!(plan_room(false, None), 0);
        assert_eq!(plan_room(false, Some(4 << 30)), 0);
    }
}
