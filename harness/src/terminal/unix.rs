    use std::ffi::c_int;
    use std::io::{self, Write};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);
    #[cfg(target_os = "macos")]
    type Flag = u64;
    #[cfg(target_os = "linux")]
    type Flag = u32;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Termios {
        input: Flag,
        output: Flag,
        control: Flag,
        local: Flag,
        #[cfg(target_os = "linux")]
        line: u8,
        #[cfg(target_os = "macos")]
        chars: [u8; 20],
        #[cfg(target_os = "linux")]
        chars: [u8; 32],
        input_speed: Flag,
        output_speed: Flag,
    }
    #[repr(C)]
    #[derive(Default)]
    struct WinSize {
        rows: u16,
        columns: u16,
        x: u16,
        y: u16,
    }
    // termios.h and sys/ioctl.h agree on these layouts for x86-64 and arm64.
    // A layout mismatch here lets libc overwrite the raw-mode restore state.
    const _: () = {
        assert!(std::mem::size_of::<WinSize>() == 8);
        assert!(std::mem::align_of::<WinSize>() == 2);
        #[cfg(target_os = "macos")]
        {
            assert!(std::mem::size_of::<Termios>() == 72);
            assert!(std::mem::offset_of!(Termios, chars) == 32);
            assert!(std::mem::offset_of!(Termios, input_speed) == 56);
            assert!(std::mem::offset_of!(Termios, output_speed) == 64);
        }
        #[cfg(target_os = "linux")]
        {
            assert!(std::mem::size_of::<Termios>() == 60);
            assert!(std::mem::offset_of!(Termios, line) == 16);
            assert!(std::mem::offset_of!(Termios, chars) == 17);
            assert!(std::mem::offset_of!(Termios, input_speed) == 52);
            assert!(std::mem::offset_of!(Termios, output_speed) == 56);
        }
    };
    unsafe extern "C" {
        fn tcgetattr(fd: c_int, state: *mut Termios) -> c_int;
        fn tcsetattr(fd: c_int, action: c_int, state: *const Termios) -> c_int;
        fn cfmakeraw(state: *mut Termios);
        fn ioctl(fd: c_int, request: usize, ...) -> c_int;
        fn signal(number: c_int, handler: usize) -> usize;
        fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    }
    extern "C" fn interrupt(_: c_int) {
        INTERRUPTED.store(true, Ordering::Relaxed);
    }
    pub struct Signals {
        saved: [(c_int, usize); 3],
    }
    impl Signals {
        pub fn install() -> Self {
            INTERRUPTED.store(false, Ordering::Relaxed);
            let mut guard = Self {
                saved: [(1, 0), (2, 0), (15, 0)],
            };
            for (number, old) in &mut guard.saved {
                *old = unsafe { signal(*number, interrupt as *const () as usize) };
            }
            guard
        }
    }
    impl Drop for Signals {
        fn drop(&mut self) {
            for (number, old) in self.saved {
                if old != usize::MAX {
                    unsafe {
                        signal(number, old);
                    }
                }
            }
        }
    }
    pub struct Guard {
        saved: Termios,
        _blocking: Blocking,
        _signals: Signals,
        active: Arc<AtomicBool>,
        previous_hook: Option<Arc<PanicHook>>,
    }
    type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static>;
    // File-status flags survive exec and are shared with the launching shell.
    // A previous terminal program may leave O_NONBLOCK set. cfmakeraw only
    // changes termios: otherwise the first idle read/write can return EAGAIN,
    // abort run_loop, and leave only the session exit receipt on screen.
    struct Blocking(Vec<(c_int, c_int)>);
    impl Blocking {
        fn enter() -> io::Result<Self> {
            let mut saved = Self(Vec::new());
            #[cfg(target_os = "macos")]
            const NONBLOCK: c_int = 4;
            #[cfg(target_os = "linux")]
            const NONBLOCK: c_int = 2048;
            for fd in [0, 1] {
                let flags = unsafe { fcntl(fd, 3) }; // F_GETFL
                if flags < 0 {
                    return Err(io::Error::last_os_error());
                }
                if flags & NONBLOCK != 0 {
                    if unsafe { fcntl(fd, 4, flags & !NONBLOCK) } < 0 {
                        return Err(io::Error::last_os_error());
                    }
                    saved.0.push((fd, flags));
                }
            }
            Ok(saved)
        }
    }
    impl Drop for Blocking {
        fn drop(&mut self) {
            for &(fd, flags) in self.0.iter().rev() {
                unsafe {
                    fcntl(fd, 4, flags);
                }
            }
        }
    }
    fn restore(saved: &Termios, active: &AtomicBool) {
        if !active.swap(false, Ordering::SeqCst) {
            return;
        }
        let _ = io::stdout()
            .write_all(b"\x1b[?2026l\x1b[r\x1b[0m\x1b[?25h\x1b[?2004l\x1b[<u\x1b[>4m\x1b[?1049l\x1b[?1007r");
        let _ = io::stdout().flush();
        unsafe {
            // Discard unread raw keystrokes before returning to canonical
            // input; they must not become commands in the caller's shell.
            const TCSAFLUSH: c_int = 2;
            tcsetattr(0, TCSAFLUSH, saved);
        }
    }
    impl Guard {
        pub fn active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
        pub fn restore(&self) {
            restore(&self.saved, &self.active);
        }
        pub fn enter(fullscreen: bool) -> io::Result<Self> {
            let blocking = Blocking::enter()?;
            // SAFETY: termios layout follows the platform ABI; tcgetattr writes
            // all fields before assume_init, and cfmakeraw receives valid state.
            let mut saved = std::mem::MaybeUninit::<Termios>::uninit();
            if unsafe { tcgetattr(0, saved.as_mut_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let saved = unsafe { saved.assume_init() };
            let mut raw = saved;
            unsafe {
                cfmakeraw(&mut raw);
            }
            if unsafe { tcsetattr(0, 0, &raw) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let active = Arc::new(AtomicBool::new(true));
            let previous_hook = Arc::new(std::panic::take_hook());
            let hook_active = active.clone();
            let hook_previous = previous_hook.clone();
            std::panic::set_hook(Box::new(move |info| {
                // Hooks run before unwinding, including worker panics. Restore
                // first so the diagnostic survives the alternate screen, then
                // stop UI redraws before delegating to the existing hook.
                INTERRUPTED.store(true, Ordering::Relaxed);
                restore(&saved, &hook_active);
                hook_previous(info);
            }));
            let guard = Self {
                saved,
                _blocking: blocking,
                _signals: Signals::install(),
                active,
                previous_hook: Some(previous_hook),
            };
            // Keep mouse selection and the terminal's clipboard available.
            // Disable alternate-scroll arrows so wheels cannot recall prompts.
            // Supported terminals distinguish modified Enter through kitty or
            // xterm keyboard reporting; legacy terminals retain Ctrl+J/Alt+Enter.
            if fullscreen {
                io::stdout().write_all(b"\x1b[?1049h\x1b[2J\x1b[H")?;
            }
            io::stdout().write_all(b"\x1b[?1007s\x1b[?1007l\x1b[?2004h\x1b[>4;2m\x1b[>1u")?;
            io::stdout().flush()?;
            Ok(guard)
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            self.restore();
            // set_hook itself panics during unwinding. A worker panic reaches
            // this drop on the main thread; a main-thread panic exits with the
            // inactive wrapper still forwarding to the original hook.
            if !std::thread::panicking() {
                drop(std::panic::take_hook());
                if let Some(previous) = self.previous_hook.take() {
                    match Arc::try_unwrap(previous) {
                        Ok(hook) => std::panic::set_hook(hook),
                        Err(hook) => std::panic::set_hook(Box::new(move |info| hook(info))),
                    }
                }
            }
        }
    }
    pub fn width() -> Option<usize> {
        dimensions().map(|(width, _)| width)
    }
    pub fn dimensions() -> Option<(usize, usize)> {
        let mut size = WinSize::default();
        #[cfg(target_os = "macos")]
        const REQUEST: usize = 0x40087468;
        #[cfg(target_os = "linux")]
        const REQUEST: usize = 0x5413;
        if unsafe { ioctl(1, REQUEST, &mut size) } == 0 && size.columns > 0 && size.rows > 0 {
            Some((size.columns as usize, size.rows as usize))
        } else {
            None
        }
    }
