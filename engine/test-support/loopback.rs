use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{
    Barrier,
    atomic::{AtomicBool, Ordering},
};

/// Retry only connection failures reported by this fixture, using fresh ports.
pub(super) fn with_ports<R>(
    count: usize,
    mut setup: impl FnMut(&[SocketAddr]) -> Result<R, Vec<String>>,
) -> R {
    const ATTEMPTS: usize = 3;
    for attempt in 1.. {
        let held: Vec<_> = (0..count)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addresses: Vec<_> = held.iter().map(|socket| socket.local_addr().unwrap()).collect();
        drop(held);

        let errors = match setup(&addresses) {
            Ok(ready) => return ready,
            Err(errors) => errors,
        };
        let port_taken = errors.iter().any(|error| {
            error.starts_with("listen:") && error.contains("Address already in use")
        });
        let rendezvous_failed = !errors.is_empty() && errors.iter().all(|error| {
            matches!(error.as_str(), "rank connection timed out" | "rank connection stopped")
                || error.starts_with("connect next rank:")
        });
        assert!(
            attempt < ATTEMPTS && (port_taken || rendezvous_failed),
            "loopback rank setup failed at attempt {attempt}: {errors:?}"
        );
        eprintln!("loopback rank setup failed at attempt {attempt}, trying fresh ports: {errors:?}");
    }
    unreachable!()
}

/// Each rank keeps its transport on its own thread. The test runs once, after
/// all ranks have connected, and any assertion panic is propagated.
pub(super) fn run_ranks<T, R: Send>(
    world: usize,
    ports_per_rank: usize,
    connect: impl Fn(usize, &[SocketAddr]) -> Result<T, String> + Sync,
    run: impl Fn(usize, T) -> R + Sync,
) -> Vec<R> {
    with_ports(world * ports_per_rank, |addresses| {
        let ready = Barrier::new(world);
        let failed = AtomicBool::new(false);
        let results = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..world)
                .map(|rank| {
                    let (connect, run, addresses, failed, ready) =
                        (&connect, &run, addresses, &failed, &ready);
                    scope.spawn(move || {
                        // Release the other ranks even if connecting panics.
                        let result = panic::catch_unwind(AssertUnwindSafe(|| connect(rank, addresses)));
                        if !matches!(&result, Ok(Ok(_))) {
                            failed.store(true, Ordering::Relaxed);
                        }
                        ready.wait();
                        match result {
                            Ok(Ok(transport)) => Ok((!failed.load(Ordering::Relaxed))
                                .then(|| run(rank, transport))),
                            Ok(Err(error)) => Err(error),
                            Err(payload) => panic::resume_unwind(payload),
                        }
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| job.join().unwrap_or_else(|payload| panic::resume_unwind(payload)))
                .collect::<Vec<_>>()
        });
        if failed.into_inner() {
            return Err(results.into_iter().filter_map(Result::err).collect());
        }
        Ok(results.into_iter().map(|result| result.unwrap().unwrap()).collect())
    })
}
