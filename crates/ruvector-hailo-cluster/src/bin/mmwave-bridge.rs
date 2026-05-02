//! `ruvector-mmwave-bridge` — host-side daemon that reads a 60 GHz
//! mmWave radar (Seeed MR60BHA2 over USB-serial) and surfaces decoded
//! vital signs.
//!
//! Iter 115 (host-side companion to iter A on the ESP32). Shares the
//! same `ruvector_mmwave::Mr60Parser` state machine that runs on the
//! ESP32-S3 firmware — exactly one tested implementation, two callers.
//!
//! Architectural fit: the radar enumerates as a `/dev/ttyUSB*` (CH340
//! / CP210x bridge variants) or `/dev/ttyACM*` (native USB-CDC variants
//! like Seeed's pre-soldered USB stick). Either way the byte stream is
//! identical Seeed mmWave protocol; this bin is the host counterpart to
//! the ESP32 firmware's UART read loop.
//!
//! # Usage
//!
//! ```text
//!   ruvector-mmwave-bridge --device /dev/ttyUSB0 [--baud 115200]
//!   ruvector-mmwave-bridge --simulator [--rate 10]   # synthesised frames @ N Hz
//!   ruvector-mmwave-bridge --auto                    # scan tty nodes for an MR60 SOF
//! ```
//!
//! Iter 116 will add `--workers <addr>` + the existing TLS/mTLS flag
//! set (`--workers-file-sig` / `--workers-file-pubkey`) so each
//! decoded vital can be posted as an embed RPC into the cluster's
//! §1b-gated path. Today's bin logs to stdout/stderr only.

use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ruvector_mmwave::{invert_xor_public, Event, Mr60Parser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut device: Option<PathBuf> = None;
    let mut auto_scan = false;
    let mut simulator = false;
    let mut sim_rate_hz: u32 = 10;
    let mut baud: u32 = 115_200;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                device = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--auto" => {
                auto_scan = true;
                i += 1;
            }
            "--simulator" => {
                simulator = true;
                i += 1;
            }
            "--rate" => {
                sim_rate_hz = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                i += 2;
            }
            "--baud" => {
                baud = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(115_200);
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--version" | "-V" => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => return Err(format!("unknown arg: {}", other).into()),
        }
    }

    // Mode selection precedence: --simulator wins (always — operator
    // explicitly asked for synthetic), then --device, then --auto.
    if simulator {
        if !quiet {
            eprintln!(
                "ruvector-mmwave-bridge: simulator mode @ {} Hz (no hardware required)",
                sim_rate_hz
            );
        }
        run_simulator(sim_rate_hz, quiet)?;
        return Ok(());
    }

    let dev = if let Some(d) = device {
        d
    } else if auto_scan {
        scan_for_radar(baud, quiet)?
    } else {
        return Err(
            "must pass exactly one of --device <path> / --simulator / --auto".into(),
        );
    };

    if !quiet {
        eprintln!(
            "ruvector-mmwave-bridge: opening {} @ {} baud",
            dev.display(),
            baud
        );
    }
    set_baud(&dev, baud)?;
    let mut file = std::fs::File::open(&dev)?;
    run_serial(&mut file, quiet)
}

/// Configure the tty's line settings to raw + N81 + the requested baud.
/// Uses `stty` because pulling in `nix` or `serialport` for a single
/// `tcsetattr` call is overkill — this bin is host-only and stty is
/// universally available where /dev/ttyUSB* + /dev/ttyACM* live.
fn set_baud(dev: &std::path::Path, baud: u32) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new("stty")
        .args([
            "-F",
            dev.to_str().ok_or("non-utf8 device path")?,
            &baud.to_string(),
            "raw",
            "-echo",
            "-echoe",
            "-echok",
            "cs8",
            "-parenb",
            "-cstopb",
            "-crtscts",
        ])
        .status()?;
    if !status.success() {
        return Err(format!("stty failed: exit {:?}", status.code()).into());
    }
    Ok(())
}

/// Drive the parser from a real serial device. Loops until EOF or
/// SIGINT (handled implicitly by std::io::Read returning Ok(0) /
/// errors). Logs decoded events to stdout one per line.
fn run_serial<R: Read>(reader: &mut R, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = Mr60Parser::new();
    let mut buf = [0u8; 256];
    let started = Instant::now();
    let mut total_events = 0u64;
    let mut last_status = Instant::now();

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            // EOF from std::fs::File means the device disappeared
            // (radar unplugged). Exit cleanly with a non-zero code so
            // a supervisor (systemd, runit) can restart on hot-plug.
            if !quiet {
                eprintln!("ruvector-mmwave-bridge: EOF on serial — radar disconnected");
            }
            return Err("device disconnected".into());
        }
        parser.feed_slice(&buf[..n], |ev| {
            total_events += 1;
            emit_event(&ev, started.elapsed());
        });
        let _ = quiet;

        // 1 Hz status when nothing has happened — keeps log scrapers
        // from thinking the bridge is wedged on a dead-radar stream.
        if last_status.elapsed() >= Duration::from_secs(1) {
            if !quiet && total_events == 0 {
                eprintln!(
                    "ruvector-mmwave-bridge: 0 events in {:?} — check radar power + baud",
                    started.elapsed()
                );
            }
            last_status = Instant::now();
        }
    }
}

/// Synthetic frame generator — bypasses real hardware. Useful for
/// (a) demoing the pipeline without the radar attached, (b) pumping
/// regression-test fixtures into a downstream consumer, (c) iter-116's
/// upcoming "post to cluster over mTLS" path which needs a stable
/// frame source for soak testing.
fn run_simulator(rate_hz: u32, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let interval = Duration::from_secs_f64(1.0 / rate_hz.max(1) as f64);
    let started = Instant::now();
    let mut parser = Mr60Parser::new();
    let mut tick: u64 = 0;
    loop {
        let frame_bytes = synthesise_frame(tick);
        // `quiet` only mutes informational stderr (the "simulator mode @
        // N Hz" banner, the periodic "0 events" warning). Decoded
        // events on stdout are always emitted — they're the bin's
        // primary output and downstream consumers (jq, log scrapers,
        // iter-116's poster) need them regardless of verbosity.
        let _ = quiet;
        parser.feed_slice(&frame_bytes, |ev| {
            emit_event(&ev, started.elapsed());
        });
        tick = tick.wrapping_add(1);
        std::thread::sleep(interval);
    }
}

/// Build a synthetic MR60BHA2 frame whose contents cycle through the
/// four interesting frame types so a downstream consumer sees the full
/// event matrix. `tick % 4` picks the type.
fn synthesise_frame(tick: u64) -> Vec<u8> {
    // Random-walk vital signs so the simulator output looks like a
    // realistic radar trace, not a constant.
    let breathing_bpm = 12 + ((tick / 4) % 8) as u8; // 12..19 bpm
    let heart_rate_bpm = 60 + ((tick * 7) % 40) as u8; // 60..99 bpm
    let distance_cm = 80 + ((tick * 13) % 200) as u16; // 80..279 cm
    let presence: u8 = if tick % 8 < 6 { 1 } else { 0 };

    let (frame_type, payload): (u16, Vec<u8>) = match tick % 4 {
        0 => (0x0A14, vec![breathing_bpm]),
        1 => (0x0A15, vec![heart_rate_bpm]),
        2 => (0x0A16, vec![(distance_cm >> 8) as u8, distance_cm as u8]),
        _ => (0x0F09, vec![presence]),
    };

    let mut header = vec![
        0x01u8,
        0x00, 0x00,
        (payload.len() >> 8) as u8, payload.len() as u8,
        (frame_type >> 8) as u8, frame_type as u8,
    ];
    let hcksum = invert_xor_public(&header);
    header.push(hcksum);
    let dcksum = invert_xor_public(&payload);
    let mut out = header;
    out.extend_from_slice(&payload);
    out.push(dcksum);
    out
}

/// Scan `/dev/ttyUSB*` + `/dev/ttyACM*` for the MR60BHA2 SOF byte
/// (`0x01`) followed by a valid header checksum. First match wins.
/// 1.5 second probe per device — enough for ~15-20 frames at the
/// MR60BHA2's typical 10 Hz output rate.
fn scan_for_radar(baud: u32, quiet: bool) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::fs;
    let mut candidates: Vec<PathBuf> = Vec::new();
    for prefix in ["/dev/ttyUSB", "/dev/ttyACM"] {
        for n in 0..16 {
            let p = PathBuf::from(format!("{}{}", prefix, n));
            if p.exists() {
                candidates.push(p);
            }
        }
    }
    if candidates.is_empty() {
        return Err("--auto: no /dev/ttyUSB* or /dev/ttyACM* nodes found".into());
    }

    for cand in candidates {
        if !quiet {
            eprintln!("ruvector-mmwave-bridge: probing {}", cand.display());
        }
        if set_baud(&cand, baud).is_err() {
            continue;
        }
        let mut f = match fs::File::open(&cand) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut parser = Mr60Parser::new();
        let mut buf = [0u8; 64];
        let deadline = Instant::now() + Duration::from_millis(1500);
        let mut got_real_event = false;
        while Instant::now() < deadline && !got_real_event {
            // Non-blocking read via a small chunk — the kernel will
            // return whatever's available.
            match f.read(&mut buf) {
                Ok(n) if n > 0 => {
                    parser.feed_slice(&buf[..n], |ev| {
                        if matches!(
                            ev,
                            Event::Breathing { .. }
                                | Event::HeartRate { .. }
                                | Event::Distance { .. }
                                | Event::Presence { .. }
                        ) {
                            got_real_event = true;
                        }
                    });
                }
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        if got_real_event {
            if !quiet {
                eprintln!(
                    "ruvector-mmwave-bridge: --auto found radar on {}",
                    cand.display()
                );
            }
            return Ok(cand);
        }
    }
    Err("--auto: no MR60BHA2-shaped frames found on any tty node within probe window".into())
}

/// Emit one decoded event as a stdout line. JSON-shaped so log
/// scrapers + iter 116's cluster-poster can both consume it cleanly.
fn emit_event(ev: &Event, t: Duration) {
    let ts_ms = t.as_millis();
    match ev {
        Event::Breathing { bpm } => println!(
            r#"{{"t_ms":{},"kind":"breathing","bpm":{}}}"#,
            ts_ms, bpm
        ),
        Event::HeartRate { bpm } => println!(
            r#"{{"t_ms":{},"kind":"heart_rate","bpm":{}}}"#,
            ts_ms, bpm
        ),
        Event::Distance { cm } => println!(
            r#"{{"t_ms":{},"kind":"distance","cm":{}}}"#,
            ts_ms, cm
        ),
        Event::Presence { present } => println!(
            r#"{{"t_ms":{},"kind":"presence","present":{}}}"#,
            ts_ms, present
        ),
        Event::Unknown { frame_type, payload_len } => println!(
            r#"{{"t_ms":{},"kind":"unknown","frame_type":"0x{:04x}","payload_len":{}}}"#,
            ts_ms, frame_type, payload_len
        ),
        Event::ChecksumError | Event::Resync => {
            // Don't pollute the stream — these surface as counter
            // increments in iter 116's status path.
        }
    }
}

fn print_help() {
    println!(
        "{} {} — host-side bridge for MR60BHA2 60 GHz mmWave radar (ADR-063)\n\
\n\
USAGE:\n    ruvector-mmwave-bridge <MODE> [OPTIONS]\n\
\n\
MODE (exactly one):\n    \
    --device <path>      Read from a specific tty (e.g. /dev/ttyUSB0).\n    \
    --auto               Scan /dev/ttyUSB* + /dev/ttyACM* for the radar.\n    \
    --simulator          Generate synthetic frames; no hardware required.\n\
\n\
OPTIONS:\n    \
    --baud <N>           UART baud (default 115200, MR60BHA2 stock).\n    \
    --rate <Hz>          Simulator frame rate (default 10).\n    \
    --quiet              Suppress informational stderr; keep stdout JSON.\n    \
    --help               This message.\n    \
    --version            Print version.\n\
\n\
OUTPUT:\n    \
    One JSON object per decoded event on stdout, e.g.:\n    \
    {{\"t_ms\":150,\"kind\":\"heart_rate\",\"bpm\":72}}\n\
\n\
Iter 116 will add --workers / --workers-file-sig / etc. for posting\n\
into the hailo-backend cluster over mTLS.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
    );
}
