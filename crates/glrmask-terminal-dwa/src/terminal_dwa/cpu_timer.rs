#[derive(Clone, Copy, Debug)]
pub struct CpuTimer {
    start_wall: std::time::Instant,
    start_cpu: libc::timespec,
}

impl CpuTimer {
    #[inline]
    pub fn start() -> Self {
        let mut start_cpu = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe {
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut start_cpu);
        }
        Self {
            start_wall: std::time::Instant::now(),
            start_cpu,
        }
    }

    #[inline]
    pub fn elapsed(&self) -> (f64, f64) {
        let wall_ms = self.start_wall.elapsed().as_secs_f64() * 1000.0;
        let mut now_cpu = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe {
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now_cpu);
        }
        let cpu_sec = (now_cpu.tv_sec - self.start_cpu.tv_sec) as f64;
        let cpu_nsec = (now_cpu.tv_nsec - self.start_cpu.tv_nsec) as f64;
        let cpu_ms = (cpu_sec * 1000.0) + (cpu_nsec / 1_000_000.0);
        (wall_ms, cpu_ms)
    }

    #[inline]
    pub fn elapsed_wall_ms(&self) -> f64 {
        self.start_wall.elapsed().as_secs_f64() * 1000.0
    }

    #[inline]
    pub fn elapsed_cpu_ms(&self) -> f64 {
        let mut now_cpu = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe {
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now_cpu);
        }
        let cpu_sec = (now_cpu.tv_sec - self.start_cpu.tv_sec) as f64;
        let cpu_nsec = (now_cpu.tv_nsec - self.start_cpu.tv_nsec) as f64;
        (cpu_sec * 1000.0) + (cpu_nsec / 1_000_000.0)
    }
}
