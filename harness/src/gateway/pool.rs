//! Bounded stdlib worker pool for gateway connections.
//! Reuses parked worker threads across requests to eliminate per-connection
//! OS thread creation overhead while preserving the caller's concurrency cap.
use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

const IDLE_KEEPALIVE: Duration = Duration::from_secs(30);

struct State<T> {
    queue: VecDeque<T>,
    idle: usize,
    total: usize,
    closed: bool,
}

pub(super) struct WorkerPool<T: Send + 'static> {
    name: &'static str,
    max_workers: usize,
    shared: Arc<(Mutex<State<T>>, Condvar)>,
    handler: Arc<dyn Fn(T) + Send + Sync>,
}

impl<T: Send + 'static> WorkerPool<T> {
    pub(super) fn new(
        name: &'static str,
        max_workers: usize,
        handler: Arc<dyn Fn(T) + Send + Sync>,
    ) -> Self {
        Self {
            name,
            max_workers: max_workers.max(1),
            shared: Arc::new((
                Mutex::new(State {
                    queue: VecDeque::new(),
                    idle: 0,
                    total: 0,
                    closed: false,
                }),
                Condvar::new(),
            )),
            handler,
        }
    }

    pub(super) fn dispatch(&self, job: T) -> io::Result<()> {
        let (lock, cvar) = &*self.shared;
        let spawn = {
            let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
            state.queue.push_back(job);
            if state.idle >= state.queue.len() {
                cvar.notify_one();
                false
            } else if state.total < self.max_workers {
                state.total += 1;
                true
            } else {
                false
            }
        };
        if spawn {
            let shared = Arc::clone(&self.shared);
            let handler = Arc::clone(&self.handler);
            if let Err(error) = thread::Builder::new().name(self.name.into()).spawn(move || {
                worker_loop(shared, handler);
            }) {
                let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
                state.total = state.total.saturating_sub(1);
                let _ = state.queue.pop_back();
                return Err(error);
            }
        }
        Ok(())
    }
}

impl<T: Send + 'static> Drop for WorkerPool<T> {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        cvar.notify_all();
    }
}

fn worker_loop<T: Send + 'static>(
    shared: Arc<(Mutex<State<T>>, Condvar)>,
    handler: Arc<dyn Fn(T) + Send + Sync>,
) {
    let (lock, cvar) = &*shared;
    loop {
        let job = {
            let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(job) = state.queue.pop_front() {
                    break Some(job);
                }
                if state.closed {
                    state.total = state.total.saturating_sub(1);
                    break None;
                }
                state.idle += 1;
                let (guard, timeout) = cvar
                    .wait_timeout(state, IDLE_KEEPALIVE)
                    .unwrap_or_else(|e| e.into_inner());
                state = guard;
                state.idle = state.idle.saturating_sub(1);
                if let Some(job) = state.queue.pop_front() {
                    break Some(job);
                }
                if state.closed || timeout.timed_out() {
                    state.total = state.total.saturating_sub(1);
                    break None;
                }
            }
        };
        let Some(job) = job else { break };
        handler(job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    #[test]
    fn reuses_worker_threads_across_sequential_jobs() {
        let (tx, rx) = mpsc::channel();
        let tx = Arc::new(Mutex::new(tx));
        let count = Arc::new(AtomicUsize::new(0));
        let count_ref = Arc::clone(&count);
        let pool = WorkerPool::new(
            "test-pool",
            4,
            Arc::new(move |n: usize| {
                count_ref.fetch_add(n, Ordering::Relaxed);
                let _ = tx.lock().unwrap().send(thread::current().id());
            }),
        );
        let mut ids = Vec::new();
        for _ in 0..8 {
            pool.dispatch(1).unwrap();
            ids.push(rx.recv_timeout(Duration::from_secs(2)).unwrap());
        }
        assert_eq!(count.load(Ordering::Relaxed), 8);
        assert!(ids.iter().all(|id| *id == ids[0]), "sequential jobs reused one worker");
    }

    #[test]
    fn dispatches_concurrent_burst_after_warmup_in_parallel() {
        let (warm_tx, warm_rx) = mpsc::channel();
        let warm_tx = Arc::new(Mutex::new(Some(warm_tx)));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let done = Arc::new(AtomicUsize::new(0));
        let barrier_ref = Arc::clone(&barrier);
        let done_ref = Arc::clone(&done);
        let pool = WorkerPool::new(
            "test-burst",
            8,
            Arc::new(move |is_warm: bool| {
                if is_warm {
                    if let Some(tx) = warm_tx.lock().unwrap().take() {
                        let _ = tx.send(());
                    }
                    return;
                }
                barrier_ref.wait();
                done_ref.fetch_add(1, Ordering::Relaxed);
            }),
        );
        // Warm one worker so state.idle == 1 before the 8-way concurrent burst.
        pool.dispatch(true).unwrap();
        warm_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        thread::sleep(Duration::from_millis(5));
        for _ in 0..8 {
            pool.dispatch(false).unwrap();
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while done.load(Ordering::Relaxed) < 8 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(done.load(Ordering::Relaxed), 8, "all 8 burst jobs must run concurrently");
    }
}
